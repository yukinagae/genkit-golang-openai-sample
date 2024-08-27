package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sync"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

const (
	provider    = "openai"
	labelPrefix = "OpenAI"
	apiKeyEnv   = "OPENAI_API_KEY"
)

var state struct {
	mu      sync.Mutex
	initted bool
	client  *goopenai.Client
}

var (
	knownCaps = map[string]ai.ModelCapabilities{
		goopenai.GPT4o:     Multimodal,
		goopenai.GPT4oMini: Multimodal,
		goopenai.GPT4Turbo: Multimodal,
		goopenai.GPT4:      BasicText,
	}

	modelsSupportingResponseFormats = []string{
		goopenai.GPT4o,     //
		goopenai.GPT4oMini, //
		goopenai.GPT4Turbo, //
	}
)

// Config is the configuration for the plugin.
type Config struct {
	// The API key to access the service.
	// If empty, the values of the environment variables OPENAI_API_KEY will be consulted.
	APIKey string
}

// Init initializes the plugin and all known models.
// After calling Init, you may call [DefineModel] to create and register any additional generative models.
// TODO: initialize embedders
func Init(ctx context.Context, cfg *Config) (err error) {
	if cfg == nil {
		cfg = &Config{}
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.initted {
		panic(provider + ".Init not called")
	}
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s.Init: %w", provider, err)
		}
	}()

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(apiKeyEnv)
		if apiKey == "" {
			return fmt.Errorf("OpenAI requires setting %s in the environment. You can get an API key at https://platform.openai.com/api-keys", apiKeyEnv)
		}
	}

	client := goopenai.NewClient(apiKey)
	state.client = client
	state.initted = true
	for model, caps := range knownCaps {
		defineModel(model, caps)
	}
	return nil
}

// DefineModel defines an unknown model with the given name.
// The second argument describes the capability of the model.
// Use [IsDefinedModel] to determine if a model is already defined.
// After [Init] is called, only the known models are defined.
func DefineModel(name string, caps *ai.ModelCapabilities) (ai.Model, error) {
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.initted {
		panic(provider + ".Init not called")
	}
	var mc ai.ModelCapabilities
	if caps == nil {
		var ok bool
		mc, ok = knownCaps[name]
		if !ok {
			return nil, fmt.Errorf("%s.DefineModel: called with unknown model %q and nil ModelCapabilities", provider, name)
		}
	} else {
		mc = *caps
	}
	return defineModel(name, mc), nil
}

// requires state.mu
func defineModel(name string, caps ai.ModelCapabilities) ai.Model {
	meta := &ai.ModelMetadata{
		Label:    labelPrefix + " - " + name,
		Supports: caps,
	}
	return ai.DefineModel(provider, name, meta, func(
		ctx context.Context,
		input *ai.GenerateRequest,
		cb func(context.Context, *ai.GenerateResponseChunk) error,
	) (*ai.GenerateResponse, error) {
		return generate(ctx, state.client, name, input, cb)
	})
}

// IsDefinedModel reports whether the named [Model] is defined by this plugin.
func IsDefinedModel(name string) bool {
	return ai.IsDefinedModel(provider, name)
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func Model(name string) ai.Model {
	return ai.LookupModel(provider, name)
}

func generate(
	ctx context.Context,
	client *goopenai.Client,
	model string,
	input *ai.GenerateRequest,
	cb func(context.Context, *ai.GenerateResponseChunk) error, // TODO: implement streaming
) (*ai.GenerateResponse, error) {
	req, err := convertRequest(model, input)
	if err != nil {
		return nil, err
	}
	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}

	jsonMode := false
	if input.Output != nil &&
		input.Output.Format == ai.OutputFormatJSON {
		jsonMode = true
	}
	r := translateResponse(resp, jsonMode)
	r.Request = input
	return r, nil
}

func convertRequest(model string, input *ai.GenerateRequest) (goopenai.ChatCompletionRequest, error) {
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return goopenai.ChatCompletionRequest{}, err
	}

	tools, err := convertTools(input.Tools)
	if err != nil {
		return goopenai.ChatCompletionRequest{}, err
	}

	chatCompletionRequest := goopenai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
		N:        input.Candidates,
	}

	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok && c != nil {
		if c.MaxOutputTokens != 0 {
			chatCompletionRequest.MaxTokens = c.MaxOutputTokens
		}
		if len(c.StopSequences) > 0 {
			chatCompletionRequest.Stop = c.StopSequences
		}
		if c.Temperature != 0 {
			chatCompletionRequest.Temperature = float32(c.Temperature)
		}
		if c.TopP != 0 {
			chatCompletionRequest.TopP = float32(c.TopP)
		}
	}

	if input.Output != nil &&
		input.Output.Format != "" &&
		slices.Contains(modelsSupportingResponseFormats, model) {
		switch input.Output.Format {
		case ai.OutputFormatJSON:
			chatCompletionRequest.ResponseFormat = &goopenai.ChatCompletionResponseFormat{
				Type: goopenai.ChatCompletionResponseFormatTypeJSONObject,
				JSONSchema: &goopenai.ChatCompletionResponseFormatJSONSchema{
					Schema: &MapJSONMarshaller{Data: input.Output.Schema},
					Strict: true,
				},
			}
		case ai.OutputFormatText:
			chatCompletionRequest.ResponseFormat = &goopenai.ChatCompletionResponseFormat{
				Type: goopenai.ChatCompletionResponseFormatTypeText,
			}
		default:
			return goopenai.ChatCompletionRequest{}, fmt.Errorf("unknown part type in a request")
		}
	}

	return chatCompletionRequest, nil
}

type MapJSONMarshaller struct {
	Data map[string]any
}

func (m *MapJSONMarshaller) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.Data)
}

func convertMessages(messages []*ai.Message) ([]goopenai.ChatCompletionMessage, error) {
	var msgs []goopenai.ChatCompletionMessage
	for _, m := range messages {
		role := fromAIRoleToOpenAIRole(m.Role)
		switch role {
		case goopenai.ChatMessageRoleUser:
			var multiContent []goopenai.ChatMessagePart
			for _, part := range m.Content {
				p, err := toOpenAiTextAndMedia(part)
				if err != nil {
					return nil, err
				}
				multiContent = append(multiContent, p)
			}
			msgs = append(msgs, goopenai.ChatCompletionMessage{
				Role:         role,
				MultiContent: multiContent,
			})
		case goopenai.ChatMessageRoleSystem:
			msgs = append(msgs, goopenai.ChatCompletionMessage{
				Role:    role,
				Content: m.Content[0].Text,
			})
		case goopenai.ChatMessageRoleAssistant:
			var toolCalls []goopenai.ToolCall
			for _, part := range m.Content {
				if !part.IsToolRequest() {
					continue
				}
				toolCalls = append(toolCalls, goopenai.ToolCall{
					ID:   part.ToolRequest.Name,
					Type: goopenai.ToolTypeFunction,
					Function: goopenai.FunctionCall{
						Name:      part.ToolRequest.Name,
						Arguments: mapToJSONString(part.ToolRequest.Input),
					},
				})
			}
			if len(toolCalls) > 0 {
				msgs = append(msgs, goopenai.ChatCompletionMessage{
					Role:      role,
					ToolCalls: toolCalls,
				})
			} else {
				msgs = append(msgs, goopenai.ChatCompletionMessage{
					Role:    role,
					Content: m.Content[0].Text,
				})
			}
		case goopenai.ChatMessageRoleTool:
			for _, part := range m.Content {
				msgs = append(msgs, goopenai.ChatCompletionMessage{
					Role:       role,
					ToolCallID: part.ToolResponse.Name,
					Content:    mapToJSONString(part.ToolResponse.Output),
					Name:       part.ToolResponse.Name,
				})
			}
		default:
			return nil, fmt.Errorf("Unknown OpenAI Role %s", role)
		}
	}
	return msgs, nil
}

func toOpenAiTextAndMedia(part *ai.Part) (goopenai.ChatMessagePart, error) {
	switch {
	case part.IsText():
		return goopenai.ChatMessagePart{
			Type: goopenai.ChatMessagePartTypeText,
			Text: part.Text,
		}, nil
	case part.IsMedia():
		return goopenai.ChatMessagePart{
			Type: goopenai.ChatMessagePartTypeImageURL,
			ImageURL: &goopenai.ChatMessageImageURL{
				URL:    part.Text,
				Detail: goopenai.ImageURLDetailAuto,
			},
		}, nil
	default:
		return goopenai.ChatMessagePart{}, fmt.Errorf("unknown part type in a request")
	}
}

func convertTools(inTools []*ai.ToolDefinition) ([]goopenai.Tool, error) {
	var outTools []goopenai.Tool
	for _, t := range inTools {
		parameters, err := mapToJSONRawMessage(t.InputSchema)
		if err != nil {
			return nil, err
		}
		fd := &goopenai.FunctionDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  parameters,
		}
		outTool := goopenai.Tool{
			Type:     goopenai.ToolTypeFunction,
			Function: fd,
		}
		outTools = append(outTools, outTool)
	}
	return outTools, nil
}

// Translate from a goopenai.ChatCompletionResponse to a ai.GenerateResponse.
func translateResponse(resp goopenai.ChatCompletionResponse, jsonMode bool) *ai.GenerateResponse {
	r := &ai.GenerateResponse{}

	for _, c := range resp.Choices {
		r.Candidates = append(r.Candidates, translateCandidate(c, jsonMode))
	}

	r.Usage = &ai.GenerationUsage{
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}
	r.Custom = resp
	return r
}

func fromAIRoleToOpenAIRole(aiRole ai.Role) string {
	switch aiRole {
	case ai.RoleUser:
		return goopenai.ChatMessageRoleUser
	case ai.RoleSystem:
		return goopenai.ChatMessageRoleSystem
	case ai.RoleModel:
		return goopenai.ChatMessageRoleAssistant
	case ai.RoleTool:
		return goopenai.ChatMessageRoleTool
	default:
		panic(fmt.Sprintf("Unknown ai.Role: %s", aiRole))
	}
}

// translateCandidate translates from a goopenai.ChatCompletionChoice to an ai.Candidate.
func translateCandidate(choice goopenai.ChatCompletionChoice, jsonMode bool) *ai.Candidate {
	c := &ai.Candidate{
		Index: choice.Index,
	}
	switch choice.FinishReason {
	case goopenai.FinishReasonStop, goopenai.FinishReasonToolCalls:
		c.FinishReason = ai.FinishReasonStop
	case goopenai.FinishReasonLength:
		c.FinishReason = ai.FinishReasonLength
	case goopenai.FinishReasonContentFilter:
		c.FinishReason = ai.FinishReasonBlocked
	case goopenai.FinishReasonFunctionCall:
		c.FinishReason = ai.FinishReasonOther
	case goopenai.FinishReasonNull:
		c.FinishReason = ai.FinishReasonUnknown
	default:
		c.FinishReason = ai.FinishReasonUnknown
	}
	m := &ai.Message{
		Role: ai.RoleModel,
	}

	// handle tool calls
	var toolRequestParts []*ai.Part
	for _, toolCall := range choice.Message.ToolCalls {
		toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  toolCall.Function.Name,
			Input: jsonStringToMap(toolCall.Function.Arguments),
		}))
	}
	if len(toolRequestParts) > 0 {
		m.Content = toolRequestParts
		c.Message = m
		return c
	}

	if jsonMode {
		m.Content = append(m.Content, ai.NewDataPart(choice.Message.Content))
	} else {
		m.Content = append(m.Content, ai.NewTextPart(choice.Message.Content))
	}

	c.Message = m
	return c
}

func jsonStringToMap(jsonString string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		panic(fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonString, err))
	}
	return result
}

func mapToJSONString(data map[string]any) string {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		panic(fmt.Errorf("failed to marshal map to JSON string: data, %#v %w", data, err))
	}
	return string(jsonBytes)
}

func mapToJSONRawMessage(data map[string]any) (json.RawMessage, error) {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal map to JSON string: data, %#v %w", data, err)
	}
	return json.RawMessage(jsonBytes), nil
}

// func mapToJSONSchema(data map[string]any) (*goopenai.ChatCompletionResponseFormatJSONSchema, error) {
// 	jsonData, err := json.Marshal(data)
// 	if err != nil {
// 		return nil, err
// 	}

// 	jsonSchema := &goopenai.ChatCompletionResponseFormatJSONSchema{}
// 	if err := json.Unmarshal(jsonData, jsonSchema); err != nil {
// 		return nil, fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonData, err)
// 	}
// 	return jsonSchema, nil
// }
