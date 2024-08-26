package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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
		goopenai.GPT4o: ai.ModelCapabilities{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
		},
		goopenai.GPT4oMini: ai.ModelCapabilities{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
		},
		goopenai.GPT4Turbo: ai.ModelCapabilities{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      true,
		},
		goopenai.GPT4: ai.ModelCapabilities{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
		},
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

	// TODO: Implement goopenai.NewClientWithConfig to allow passing additional configuration options.
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
	var parts []goopenai.ChatCompletionMessage
	for _, m := range input.Messages {
		ps, err := convertParts(m.Content)
		if err != nil {
			return nil, err
		}
		parts = append(parts, ps...)
	}

	tools, err := convertTools(input.Tools)
	if err != nil {
		return nil, err
	}

	chatCompletionRequest := goopenai.ChatCompletionRequest{
		Model:    model,
		Messages: parts,
		Tools:    tools,
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
		if c.TopK != 0 {
			chatCompletionRequest.N = c.TopK
		}
		if c.TopP != 0 {
			chatCompletionRequest.TopP = float32(c.TopP)
		}
	}

	resp, err := client.CreateChatCompletion(ctx, chatCompletionRequest)
	if err != nil {
		return nil, err
	}
	r := translateResponse(resp)
	r.Request = input
	return r, nil
}

func convertTools(inTools []*ai.ToolDefinition) ([]goopenai.Tool, error) {
	var outTools []goopenai.Tool
	for _, t := range inTools {
		inputSchema := mapToJSONString(t.InputSchema) // TODO: resolve $ref
		fd := &goopenai.FunctionDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  inputSchema,
		}
		outTool := goopenai.Tool{
			Type:     goopenai.ToolTypeFunction,
			Function: fd,
		}
		outTools = append(outTools, outTool)
	}
	return outTools, nil
}

// convertParts converts a slice of *ai.Part to a slice of goopenai.ChatCompletionMessage.
func convertParts(parts []*ai.Part) ([]goopenai.ChatCompletionMessage, error) {
	res := make([]goopenai.ChatCompletionMessage, 0, len(parts))
	for _, p := range parts {
		part, err := convertPart(p)
		if err != nil {
			return nil, err
		}
		res = append(res, part)
	}
	return res, nil
}

// convertPart converts a *ai.Part to a goopenai.ChatCompletionMessage.
func convertPart(p *ai.Part) (goopenai.ChatCompletionMessage, error) {
	switch {
	case p.IsText():
		return goopenai.ChatCompletionMessage{
			Role:    goopenai.ChatMessageRoleUser,
			Content: p.Text,
		}, nil
	case p.IsMedia():
		panic("not yet implemented") // TODO: implement Media part
	case p.IsData():
		panic(fmt.Sprintf("%s does not support Data parts", provider))
	case p.IsToolResponse():
		return goopenai.ChatCompletionMessage{
			Role:    goopenai.ChatMessageRoleAssistant,
			Content: mapToJSONString(p.ToolResponse.Output),
			Name:    p.ToolResponse.Name,
		}, nil
	case p.IsToolRequest():
		return goopenai.ChatCompletionMessage{
			Role:    goopenai.ChatMessageRoleAssistant,
			Content: mapToJSONString(p.ToolRequest.Input),
			Name:    p.ToolRequest.Name,
		}, nil
	default:
		panic("unknown part type in a request")
	}
}

// Translate from a goopenai.ChatCompletionResponse to a ai.GenerateResponse.
func translateResponse(resp goopenai.ChatCompletionResponse) *ai.GenerateResponse {
	r := &ai.GenerateResponse{
		Usage: &ai.GenerationUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
	}
	for _, c := range resp.Choices {
		r.Candidates = append(r.Candidates, translateCandidate(c))
	}
	return r
}

func toOpenAIRole(role string) ai.Role {
	switch role {
	case goopenai.ChatMessageRoleUser:
		return ai.RoleUser
	case goopenai.ChatMessageRoleAssistant:
		return ai.RoleModel
	case goopenai.ChatMessageRoleSystem:
		return ai.RoleSystem
	case goopenai.ChatMessageRoleTool, goopenai.ChatMessageRoleFunction:
		return ai.RoleTool
	default:
		panic(fmt.Sprintf("unknown role: %s", role))
	}
}

// translateCandidate translates from a goopenai.ChatCompletionChoice to an ai.Candidate.
func translateCandidate(choice goopenai.ChatCompletionChoice) *ai.Candidate {
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
		Role: toOpenAIRole(choice.Message.Role),
	}

	// handle single content
	if choice.Message.Content != "" {
		m.Content = append(m.Content, ai.NewTextPart(choice.Message.Content))
		c.Message = m
		return c
	}

	// handle multi-content
	for _, part := range choice.Message.MultiContent {
		switch {
		case part.Type == goopenai.ChatMessagePartTypeText:
			p := ai.NewTextPart(part.Text)
			m.Content = append(m.Content, p)
		case part.Type == goopenai.ChatMessagePartTypeImageURL:
			panic("not yet implemented") // TODO: implement Media part
		case choice.Message.ToolCalls != nil:
			for _, toolCall := range choice.Message.ToolCalls {
				p := ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  toolCall.Function.Name,
					Input: jsonStringToMap(toolCall.Function.Arguments),
				})
				m.Content = append(m.Content, p)
			}
			continue
		default:
			panic(fmt.Sprintf("unknown part %#v", part))
		}
	}
	c.Message = m
	return c
}

func jsonStringToMap(jsonString string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		panic(fmt.Sprintf("unmarshal failed to parse json string %s: %w", jsonString, err))
	}
	return result
}

func mapToJSONString(data map[string]any) string {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		panic(fmt.Errorf("failed to marshal map to JSON string: data, %v %w", data, err))
	}
	return string(jsonBytes)
}
