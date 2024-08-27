package main

import (
	"context"
	"fmt"
	"log"

	// Import the Genkit core libraries.
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	// Import the Google AI plugin.
	"github.com/firebase/genkit/go/plugins/googleai"
	// Import the OpenAI plugin.
	"github.com/yukinagae/genkit-golang-openai-sample/plugins/openai"
)

func main() {
	ctx := context.Background()

	// Initialize the Google AI plugin. When you pass an empty string for the
	// apiKey parameter, the Google AI plugin will use the value from the
	// GOOGLE_GENAI_API_KEY environment variable, which is the recommended
	// practice.
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// Initialize the OpenAI plugin. When you pass an empty string for the
	// apiKey parameter, the OpenAI plugin will use the value from the
	// OPENAI_API_KEY environment variable, which is the recommended
	// practice.
	if err := openai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// Define a simple flow that prompts an LLM to generate menu suggestions.
	genkit.DefineFlow("menuSuggestionFlow", func(ctx context.Context, input string) (string, error) {
		// The Google AI API provides access to several generative models. Here,
		// we specify gemini-1.5-flash.
		// m := googleai.Model("gemini-1.5-flash")

		// lookup OpenAI model
		m := openai.Model("gpt-4o-mini") // or "gpt-4o-mini"

		// Construct a request and send it to the model API.
		resp, err := m.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage(fmt.Sprintf(`Suggest an item for the menu of a %s themed restaurant`, input))),
			nil)
		if err != nil {
			return "", err
		}

		// Handle the response from the model API. In this sample, we just
		// convert it to a string, but more complicated flows might coerce the
		// response into structured output or chain the response into another
		// LLM call, etc.
		text := resp.Text()
		return text, nil
	})

	// Initialize Genkit and start a flow server. This call must come last,
	// after all of your plug-in configuration and flow definitions. When you
	// pass a nil configuration to Init, Genkit starts a local flow server,
	// which you can interact with using the developer UI.
	if err := genkit.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}
}
