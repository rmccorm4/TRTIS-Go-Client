package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/rmccorm4/trtis-go-client/nvidia_inferenceserver"
	"google.golang.org/grpc"
)

func main() {
	endpoint := "10.33.1.25:8001"
	// Connect to gRPC server
	conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", endpoint, err)
	}
	defer conn.Close()
	// Create client from gRPC server connection
	client := nvidia_inferenceserver.NewGRPCServiceClient(conn)
	// Create context for our request
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Request the status of the server
	response, err := client.Status(ctx, &nvidia_inferenceserver.StatusRequest{})
	if err != nil {
		log.Fatalf("Couldn't get server status: %v", err)
	}
	fmt.Printf("%T\n %v\n", response, response)
}
