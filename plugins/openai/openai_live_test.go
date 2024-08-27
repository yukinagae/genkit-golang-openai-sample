package openai_test

import (
	"context"
	"flag"
	"math"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/yukinagae/genkit-golang-openai-sample/plugins/openai"
)

// The tests here only work with an API key set to a valid value.
var apiKey = flag.String("key", "", "OpenAI API key")

// We can't test the DefineAll functions along with the other tests because
// we get duplicate definitions of models.
var testAll = flag.Bool("all", false, "test DefineAllXXX functions")

func TestLive(t *testing.T) {
	if *apiKey == "" {
		t.Skipf("no -key provided")
	}
	if *testAll {
		t.Skip("-all provided")
	}
	ctx := context.Background()
	if err := openai.Init(ctx, &openai.Config{APIKey: *apiKey}); err != nil {
		t.Fatal(err)
	}

	model := openai.Model("gpt-4o-mini")

	t.Run("generate", func(t *testing.T) {
		resp, err := ai.Generate(
			ctx,                  //
			model,                //
			ai.WithCandidates(1), //
			ai.WithTextPrompt("Just the country name where Napoleon was emperor, no period."), //
		)
		if err != nil {
			t.Fatal(err)
		}
		out := resp.Candidates[0].Message.Content[0].Text
		const want = "France"
		if out != want {
			t.Errorf("got %q, expecting %q", out, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 || resp.Usage.TotalTokens == 0 {
			t.Errorf("Empty usage stats %#v", *resp.Usage)
		}
	})

	t.Run("tool", func(t *testing.T) {
		gablorkenTool := ai.DefineTool("gablorken", "use when need to calculate a gablorken",
			func(ctx context.Context, input struct {
				Value float64
				Over  float64
			}) (float64, error) {
				return math.Pow(input.Value, input.Over), nil
			},
		)
		resp, err := ai.Generate(
			ctx,                  //
			model,                //
			ai.WithCandidates(1), //
			ai.WithTextPrompt("what is a gablorken of 2 over 3.5?"), //
			ai.WithTools(gablorkenTool),                             //
		)
		if err != nil {
			t.Fatal(err)
		}
		out := resp.Candidates[0].Message.Content[0].Text
		const want = "12.25"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})
}
