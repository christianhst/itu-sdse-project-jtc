package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Set up Python environment and mount project directory
	python := client.Container().
		From("python:3.11-bookworm").
		WithDirectory("/project", client.Host().Directory("..")).
		WithWorkdir("/project").
		WithExec([]string{"python", "--version"})

	// Install dependencies
	python = python.WithExec([]string{
		"pip", "install", "--no-cache-dir", "-r", "requirements.txt",
	})

	// Change working directory to the model folder
	python = python.WithWorkdir("/project/customer_classification_model")

	// Run preprocessing script
	python = python.WithExec([]string{
		"python", "modeling/preprocessing.py",
	})

	// Run training script
	python = python.WithExec([]string{
		"python", "modeling/train.py",
	})

	// Run model_selection script
	python = python.WithExec([]string{
		"python", "modeling/model_selection.py",
	})

	// Run deployment script
	python = python.WithExec([]string{
		"python", "modeling/deploy_model.py",
	})

	// Clean up unnecessary files
	python = python.WithExec([]string{
		"find", ".", "-name", ".DS_Store", "-delete",
	})

	// Export artifacts
	_, err = python.
		Directory("/project/customer_classification_model/artifacts").
		Export(ctx, "output/artifacts")
	if err != nil {
		return err
	}

	return nil
}
