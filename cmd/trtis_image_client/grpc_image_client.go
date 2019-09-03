package main

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"time"

	"github.com/rmccorm4/trtis-go-client/nvidia_inferenceserver"
	"google.golang.org/grpc"
)

type Flags struct {
	Verbose      bool
	Async        bool
	Streaming    bool
	ModelName    string
	ModelVersion int
	BatchSize    int
	NumClasses   int
	Scaling      string
	URL          string
	ImagePath    string
}

type ModelConfig struct {
	InputName   string
	OutputName  string
	Channels    int64
	Height      int64
	Width       int64
	InputFormat nvidia_inferenceserver.ModelInput_Format
	DataType    nvidia_inferenceserver.DataType
}

func ParseModel(status *nvidia_inferenceserver.StatusResponse, model_name string, batch_size int32, verbose bool) (ModelConfig, error) {
	var model_config ModelConfig
	server_status := status.ServerStatus
	if _, ok := server_status.ModelStatus[model_name]; !ok {
		return model_config, fmt.Errorf("Couldn't find model in server status: %s", model_name)
	}
	fmt.Printf("Found model: %s\n", model_name)
	model_status := server_status.ModelStatus[model_name]
	//fmt.Println(model_status)

	config := model_status.Config

	if len(config.Input) != 1 {
		return model_config, fmt.Errorf("Expecting 1 input, got %d", len(config.Input))
	}
	if len(config.Output) != 1 {
		return model_config, fmt.Errorf("Expecting 1 output, got %d", len(config.Output))
	}

	input := config.Input[0]
	output := config.Output[0]

	if output.DataType != nvidia_inferenceserver.DataType_TYPE_FP32 {
		return model_config, fmt.Errorf("Expecting output type to be TYPE_FP32, got %v", output.DataType)
	}

	// Output is expected to be a vector.
	// Allow any number of dimensions as long as all but 1 are size 1.
	// e.g. { 10 }, { 1, 10 }, { 10, 1, 1 } are all valid
	var non_one_cnt int
	for dim := range output.Dims {
		if dim > 1 {
			non_one_cnt += 1
			if non_one_cnt > 1 {
				return model_config, fmt.Errorf("Expecting model output to be a vector.")
			}
		}
	}

	// Model specifying maximum batch size of 0 indicates that batching
	// is not supported and so the input tensors do not expect an "N"
	// dimension (and 'batch_size' should be 1 so that only a single
	// image instance is inferred at a time).
	max_batch_size := config.MaxBatchSize
	if max_batch_size == 0 {
		if batch_size != 1 {
			return model_config, fmt.Errorf("Batching not supported for model: %s", model_name)
		}
	} else {
		if batch_size > max_batch_size {
			return model_config, fmt.Errorf("Expecting batch_size <= %d for model [%s], got %d", max_batch_size, model_name, batch_size)
		}
	}

	// Model input must have 3 dims, either CHW or HWC
	if len(input.Dims) != 3 {
		return model_config, fmt.Errorf("Expecting input to have 3 dimensions. Model [%s] input had %d", model_name, len(input.Dims))
	}

	fmt.Printf("%T\n", input.Format)
	var h, w, c int64
	if input.Format == nvidia_inferenceserver.ModelInput_FORMAT_NHWC {
		h = input.Dims[0]
		w = input.Dims[1]
		c = input.Dims[2]
	} else {
		c = input.Dims[0]
		h = input.Dims[1]
		w = input.Dims[2]

	}

	model_config = ModelConfig{
		InputName:   input.Name,
		OutputName:  output.Name,
		Channels:    c,
		Height:      h,
		Width:       w,
		InputFormat: input.Format,
		DataType:    output.DataType,
	}
	return model_config, nil
}

func requestGenerator(mc ModelConfig, FLAGS Flags) {
	request := nvidia_inferenceserver.InferRequest()
	request.ModelName = FLAGS.ModelName
	if FLAGS.ModelVersion == 0 {
		// Choose latest version
		request.ModelVersion = -1
	} else {
		request.ModelVersion = FLAGS.ModelVersion
	}

	request.MetaData.BatchSize = FLAGS.BatchSize

	output_message := nvidia_inferenceserver.InferRequestHeader.Output()
	output_message.Name = mc.OutputName
	output_message.Cls.Count = FLAGS.NumClasses
	request.MetaData.Output = append(request.MetaData.Output, output_message)

	var filenames []string
	fi, err := os.Stat(FLAGS.ImagePath)
	if err != nil {
		// TODO maybe return error instead, not sure what's better.
		log.Fatalf("Error with os.Stat for file %s: %v", FLAGS.ImagePath, err)
		os.Exit(1)
	}
	switch mode := fi.Mode(); mode {
	// ImagePath is a directory
	case mode.IsDir():
		files, err := ioutil.ReadDir(FLAGS.ImagePath)
		if err != nil {
			log.Fatal(err)
		}

		for _, f := range files {
			filenames = append(filenames, f.Name())
		}
		sort.Strings(filenames)

	// ImagePath is a file
	case mode.IsRegular():
		filenames = []string{FLAGS.ImagePath}
	default:
		log.Fatal("This probably shouldn't happen.")
	}

	// TODO: Implement python's yield/generator behavior with channels

}

func parseFlags() Flags {
	var flags Flags
	flag.BoolVar(&flags.Verbose, "v", false, "Enable verbose output.")
	flag.BoolVar(&flags.Async, "a", false, "Use asynchronous inference API")
	flag.BoolVar(&flags.Streaming, "streaming", false, "Use streaming inference API")
	flag.StringVar(&flags.ModelName, "m", "", "Name of model being served. (Required)")
	flag.IntVar(&flags.ModelVersion, "x", 0, "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "b", 1, "Batch size. Default is 1.")
	flag.IntVar(&flags.NumClasses, "c", 1, "Number of class predictions to report. Default: 1.")
	flag.StringVar(&flags.Scaling, "s", "NONE", "Type of scaling to apply to image pixels. Default: NONE")
	flag.StringVar(&flags.URL, "u", "localhost:8001", "Inference Server URL. Default: localhost:8001")
	flag.StringVar(&flags.ImagePath, "i", "", "Input image/directory. (Required)")
	flag.Parse()
	return flags
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	// Debug Defaults
	FLAGS.URL = "10.33.1.25:8001"
	FLAGS.ModelName = "resnet50_netdef"
	FLAGS.BatchSize = 1
	FLAGS.Verbose = true

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
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
	//fmt.Printf("%T\n %v\n", response, response)

	// Parse model config from status
	model_config, err := ParseModel(response, FLAGS.ModelName, int32(FLAGS.BatchSize), FLAGS.Verbose)
	if err != nil {
		log.Fatalf("Couldn't parse model %s: %v", FLAGS.ModelName, err)
	}
	fmt.Println(model_config)
}
