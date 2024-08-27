# Default target: print this help message
.PHONY: help
.DEFAULT_GOAL := help
help:
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/  /'

## genkit: Run Genkit locally
.PHONY: genkit
genkit:
	genkit start -o

## test: Test the Go modules within this package
.PHONY: test
test:
	go test -v ./...

## tidy: Tidy modfiles, format and lint .go files
.PHONY: tidy
tidy:
	go mod tidy -v
	go fmt ./...
	golangci-lint run
