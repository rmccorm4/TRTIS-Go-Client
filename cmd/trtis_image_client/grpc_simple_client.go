package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	trtis "github.com/rmccorm4/trtis-go-client/nvidia_inferenceserver"
	"google.golang.org/grpc"
)

type Flags struct {
	Verbose      bool
	ModelName    string
	ModelVersion int64
	BatchSize    int
	NumClasses   uint
	URL          string
}

type ModelConfig struct {
	InputName  string
	OutputName string
	Channels   int64
	Height     int64
	Width      int64
	DataType   trtis.DataType
}

func parseFlags() Flags {
	var flags Flags
	flag.BoolVar(&flags.Verbose, "v", false, "Enable verbose output.")
	flag.StringVar(&flags.ModelName, "m", "", "Name of model being served. (Required)")
	flag.Int64Var(&flags.ModelVersion, "x", 0, "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "b", 1, "Batch size. Default is 1.")
	flag.UintVar(&flags.NumClasses, "c", 1, "Number of class predictions to report. Default: 1.")
	flag.StringVar(&flags.URL, "u", "localhost:8001", "Inference Server URL. Default: localhost:8001")
	flag.Parse()
	return flags
}

// mode should be either "live" or "ready"
func HealthRequest(client trtis.GRPCServiceClient, mode string) *trtis.HealthResponse {
	// Create context for our request
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create health request for given mode {"live", "ready"}
	healthRequest := trtis.HealthRequest{
		Mode: mode,
	}
	// Request the status of the server
	healthResponse, err := client.Health(ctx, &healthRequest)
	if err != nil {
		log.Fatalf("Couldn't get server status: %v", err)
	}
	return healthResponse
}

func StatusRequest(client trtis.GRPCServiceClient, modelName string) *trtis.StatusResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	// Request the status of the server
	statusRequest := trtis.StatusRequest{
		ModelName: modelName,
	}
	statusResponse, err := client.Status(ctx, &statusRequest)
	if err != nil {
		log.Fatalf("Couldn't get server status: %v", err)
	}
	return statusResponse
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	// Debug Defaults
	FLAGS.URL = "10.33.1.25:8001"
	FLAGS.ModelName = "simple"

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	// Create client from gRPC server connection
	client := trtis.NewGRPCServiceClient(conn)

	liveHealthResponse := HealthRequest(client, "live")
	fmt.Printf("TRTIS Health - Live: %v\n", liveHealthResponse.Health)

	readyHealthResponse := HealthRequest(client, "ready")
	fmt.Printf("TRTIS Health - Ready: %v\n", readyHealthResponse.Health)

	statusResponse := StatusRequest(client, FLAGS.ModelName)
	fmt.Println(statusResponse)
}
