package main

import (
	"fmt"
	"image"
	"log"

	"github.com/disintegration/imaging"
)

func main() {
	// wget https://raw.githubusercontent.com/NVIDIA/tensorrt-inference-server/master/qa/images/mug.jpg
	src, err := imaging.Open("mug.jpg")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	// Resize the cropped image to width = 200px preserving the aspect ratio.
	src = imaging.Resize(src, 224, 224, imaging.Lanczos)
	// Can't access src.Pix directly from interface, need to assert underlying type
	src_nrgba := src.(*image.NRGBA)
	fmt.Printf("%T\n", src)
	fmt.Printf("%T\n", src_nrgba)
	pixels := src_nrgba.Pix
	fmt.Printf("%T\n", pixels)
	fmt.Println(len(pixels))
	//ioutil.WriteFile("gobytes.jpg", pixels, 0644)
}
