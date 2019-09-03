package main

import (
	"fmt"
	"image"
	"io/ioutil"
	"log"

	"github.com/disintegration/imaging"
)

func main() {
	// wget https://raw.githubusercontent.com/NVIDIA/tensorrt-inference-server/master/qa/images/mug.jpg
	src, err := imaging.Open("mug.jpg")
	if err != nil {
		log.Fatalf("failed to open image: %v", err)
	}

	height, width := 224, 224
	src = imaging.Resize(src, height, width, imaging.Linear)
	// Can't access src.Pix directly from interface, need to assert underlying type
	src_nrgba := src.(*image.NRGBA)
	fmt.Printf("%T\n", src)
	fmt.Printf("%T\n", src_nrgba)
	pixels := src_nrgba.Pix
	rgb_pixels := pixels[:3*height*width]
	fmt.Printf("%T\n", rgb_pixels)
	fmt.Println(len(rgb_pixels))
	ioutil.WriteFile("gobytes.jpg", rgb_pixels, 0644)
}
