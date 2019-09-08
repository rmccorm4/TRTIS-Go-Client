package main

import (
	"bytes"
	"context"
	"encoding/binary"
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
	flag.Int64Var(&flags.ModelVersion, "x", -1, "Version of model. Default: Latest Version.")
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
		log.Fatalf("Couldn't get server health: %v", err)
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

func InferRequest(client trtis.GRPCServiceClient, rawInput [][]byte, modelName string, modelVersion int64, batchSize int) *trtis.InferResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	/* We use a simple model that takes 2 input tensors of 16 integers
	each and returns 2 output tensors of 16 integers each. One
	output tensor is the element-wise sum of the inputs and one
	output is the element-wise difference. */
	inferRequestHeader := &trtis.InferRequestHeader{
		Input: []*trtis.InferRequestHeader_Input{
			&trtis.InferRequestHeader_Input{
				Name: "INPUT0",
			},
			&trtis.InferRequestHeader_Input{
				Name: "INPUT1",
			},
		},
		Output: []*trtis.InferRequestHeader_Output{
			&trtis.InferRequestHeader_Output{
				Name: "OUTPUT0",
			},
			&trtis.InferRequestHeader_Output{
				Name: "OUTPUT1",
			},
		},
		BatchSize: uint32(batchSize),
	}

	// Request the status of the server
	inferRequest := trtis.InferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		MetaData:     inferRequestHeader,
		RawInput:     rawInput,
	}

	inferResponse, err := client.Infer(ctx, &inferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return inferResponse
}

func Preprocess(inputs [][]uint32) [][]byte {
	inputData0 := inputs[0]
	inputData1 := inputs[1]

	var inputBytes0 []byte
	var inputBytes1 []byte
	// Temp variable to hold our converted int32 -> []byte
	bs := make([]byte, 4)
	inputSize := 16
	for i := 0; i < inputSize; i++ {
		binary.LittleEndian.PutUint32(bs, inputData0[i])
		inputBytes0 = append(inputBytes0, bs...)
		binary.LittleEndian.PutUint32(bs, inputData1[i])
		inputBytes1 = append(inputBytes1, bs...)
	}

	return [][]byte{inputBytes0, inputBytes1}
}

// Convert slice of 4 bytes to int32 (Assumes Little Endian)
func readInt32(fourBytes []byte) int32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval int32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}

func Postprocess(inferResponse *trtis.InferResponse) [][]int32 {
	var outputs [][]byte
	outputs = inferResponse.RawOutput
	outputBytes0 := outputs[0]
	outputBytes1 := outputs[1]

	outputSize := 16
	outputData0 := make([]int32, outputSize)
	outputData1 := make([]int32, outputSize)
	for i := 0; i < outputSize; i++ {
		outputData0[i] = readInt32(outputBytes0[i*4 : i*4+4])
		outputData1[i] = readInt32(outputBytes1[i*4 : i*4+4])
	}
	return [][]int32{outputData0, outputData1}
}

func main() {
	FLAGS := parseFlags()

	// Debug Defaults
	FLAGS.URL = "10.33.1.25:8001"
	FLAGS.ModelName = "simple"
	fmt.Println("FLAGS:", FLAGS)

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

	inputSize := 16
	inputData0 := make([]uint32, inputSize)
	inputData1 := make([]uint32, inputSize)
	for i := 0; i < inputSize; i++ {
		inputData0[i] = uint32(i)
		inputData1[i] = 1
	}
	inputs := [][]uint32{inputData0, inputData1}
	rawInput := Preprocess(inputs)

	inferResponse := InferRequest(client, rawInput, FLAGS.ModelName, FLAGS.ModelVersion, FLAGS.BatchSize)

	outputs := Postprocess(inferResponse)
	outputData0 := outputs[0]
	outputData1 := outputs[1]

	fmt.Println("\nChecking Inference Outputs\n--------------------------")
	for i := 0; i < 16; i++ {
		fmt.Printf("%d + %d = %d\n", inputData0[i], inputData1[i], outputData0[i])
		fmt.Printf("%d - %d = %d\n", inputData0[i], inputData1[i], outputData1[i])
	}
}
