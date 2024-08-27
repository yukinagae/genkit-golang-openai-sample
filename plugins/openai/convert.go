package openai

import (
	"fmt"
	"slices"

	"github.com/firebase/genkit/go/ai"
	goopenai "github.com/sashabaranov/go-openai"
)

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

func convertMessages(messages []*ai.Message) ([]goopenai.ChatCompletionMessage, error) {
	var msgs []goopenai.ChatCompletionMessage
	for _, m := range messages {
		role := fromAIRoleToOpenAIRole(m.Role)
		switch role {
		case goopenai.ChatMessageRoleUser:
			var multiContent []goopenai.ChatMessagePart
			for _, part := range m.Content {
				p, err := convertPart(part)
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

func convertPart(part *ai.Part) (goopenai.ChatMessagePart, error) {
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