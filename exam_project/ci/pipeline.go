package main

import (
	"context"
	"fmt"
	"os"

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
	// Define project and model directories
	const (
		projectDir = "/project/exam_project"
		modelDir   = projectDir + "/customer_classification_model"
	)

	// Initialize Dagger client
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		return err
	}
	defer client.Close()

	// Set up Python environment and mount project directory
	python := client.Container().
		From("python:3.11-bookworm").
		WithDirectory("/project", client.Host().Directory("../..")).
		WithWorkdir(projectDir).
		WithExec([]string{"python", "--version"})

	// Install dependencies
	python = python.WithExec([]string{
		"pip", "install", "--no-cache-dir", "-r", "requirements.txt",
	})

	// Change working directory to the model folder
	python = python.WithWorkdir(modelDir)

	python = python.WithExec([]string{"dvc", "update", "artifacts/raw_data.csv.dvc"})

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
		"python", "modeling/deploy.py",
	})

	// Clean up unnecessary files
	python = python.WithExec([]string{
		"find", ".", "-name", ".DS_Store", "-delete",
	})

	// Export artifacts
	_, err = python.
		Directory(modelDir+"/artifacts").
		Export(ctx, "output/artifacts")
	if err != nil {
		return err
	}

	return nil
}
