package openai

import (
	"context"
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
