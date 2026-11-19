// NVIDIA EdgeVision Core Script
/**
 * NVIDIA Vision Lab - High Performance Edge Detection Algorithms
 * Optimized for real-time edge processing and computer vision pipelines
 */

class EdgeDetector {
  constructor() {
    this.width = 0;
    this.height = 0;
    this.grayBuffer = null;
    this.blurBuffer = null;
    this.gxBuffer = null;
    this.gyBuffer = null;
    this.magBuffer = null;
    this.nmsBuffer = null;
    this.edgeBuffer = null;
  }

  resizeBuffers(width, height) {
    if (this.width === width && this.height === height && this.grayBuffer) return;
    this.width = width;
    this.height = height;
    const size = width * height;
    this.grayBuffer = new Float32Array(size);
    this.blurBuffer = new Float32Array(size);
    this.gxBuffer = new Float32Array(size);
    this.gyBuffer = new Float32Array(size);
    this.magBuffer = new Float32Array(size);
    this.nmsBuffer = new Float32Array(size);
    this.edgeBuffer = new Uint8Array(size);
  }

  toGrayscale(imageData) {
    const data = imageData.data;
    const gray = this.grayBuffer;
    const len = this.width * this.height;
    for (let i = 0, j = 0; i < len; i++, j += 4) {
      // Rec. 709 luminance weights
      gray[i] = 0.2126 * data[j] + 0.7152 * data[j + 1] + 0.0722 * data[j + 2];
    }
  }

  gaussianBlur(sigma = 1.4) {
    const w = this.width;
    const h = this.height;
    const src = this.grayBuffer;
    const dst = this.blurBuffer;

    // Fast 5x5 separable kernel approximation (Gaussian sigma ~ 1.4)
    // Kernel weights: [1, 4, 6, 4, 1] / 16
    const temp = new Float32Array(w * h);

    // Horizontal pass
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - 2);
        const x1 = Math.max(0, x - 1);
        const x2 = x;
        const x3 = Math.min(w - 1, x + 1);
        const x4 = Math.min(w - 1, x + 2);

        temp[rowOffset + x] = (
          src[rowOffset + x0] * 1 +
          src[rowOffset + x1] * 4 +
          src[rowOffset + x2] * 6 +
          src[rowOffset + x3] * 4 +
          src[rowOffset + x4] * 1
        ) / 16;
      }
    }

    // Vertical pass
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - 2) * w + x;
        const y1 = Math.max(0, y - 1) * w + x;
        const y2 = y * w + x;
        const y3 = Math.min(h - 1, y + 1) * w + x;
        const y4 = Math.min(h - 1, y + 2) * w + x;

        dst[y2] = (
          temp[y0] * 1 +
          temp[y1] * 4 +
          temp[y2] * 6 +
          temp[y3] * 4 +
          temp[y4] * 1
        ) / 16;
      }
    }
  }

  // Sobel Operator (Gx, Gy, Magnitude)
  sobel(options = {}) {
    const w = this.width;
    const h = this.height;
    const src = options.blur ? this.blurBuffer : this.grayBuffer;
    const gx = this.gxBuffer;
    const gy = this.gyBuffer;
    const mag = this.magBuffer;
    const threshold = options.threshold || 50;

    let maxMag = 0;

    for (let y = 1; y < h - 1; y++) {
      const prevRow = (y - 1) * w;
      const currRow = y * w;
      const nextRow = (y + 1) * w;

      for (let x = 1; x < w - 1; x++) {
        // Gx kernel: [-1 0 1; -2 0 2; -1 0 1]
        const valGx = (
          -1 * src[prevRow + x - 1] + 1 * src[prevRow + x + 1] +
          -2 * src[currRow + x - 1] + 2 * src[currRow + x + 1] +
          -1 * src[nextRow + x - 1] + 1 * src[nextRow + x + 1]
        );

        // Gy kernel: [-1 -2 -1; 0 0 0; 1 2 1]
        const valGy = (
          -1 * src[prevRow + x - 1] - 2 * src[prevRow + x] - 1 * src[prevRow + x + 1] +
           1 * src[nextRow + x - 1] + 2 * src[nextRow + x] + 1 * src[nextRow + x + 1]
        );

        const idx = currRow + x;
        gx[idx] = valGx;
        gy[idx] = valGy;
        const m = Math.sqrt(valGx * valGx + valGy * valGy);
        mag[idx] = m;
        if (m > maxMag) maxMag = m;
      }
    }

    // Threshold & binary / grayscale edge output
    const edges = this.edgeBuffer;
    edges.fill(0);
    const scale = maxMag > 0 ? 255 / maxMag : 1;

    for (let i = 0; i < w * h; i++) {
      if (options.binary) {
        edges[i] = mag[i] >= threshold ? 255 : 0;
      } else {
        const val = mag[i] * scale * (options.sensitivity || 1.2);
        edges[i] = val >= threshold ? Math.min(255, val) : 0;
      }
    }

    return edges;
  }

  // Prewitt Operator
  prewitt(options = {}) {
    const w = this.width;
    const h = this.height;
    const src = options.blur ? this.blurBuffer : this.grayBuffer;
    const mag = this.magBuffer;
    const threshold = options.threshold || 45;

    for (let y = 1; y < h - 1; y++) {
      const prevRow = (y - 1) * w;
      const currRow = y * w;
      const nextRow = (y + 1) * w;

      for (let x = 1; x < w - 1; x++) {
        // Gx: [-1 0 1; -1 0 1; -1 0 1]
        const valGx = (
          -1 * src[prevRow + x - 1] + 1 * src[prevRow + x + 1] +
          -1 * src[currRow + x - 1] + 1 * src[currRow + x + 1] +
          -1 * src[nextRow + x - 1] + 1 * src[nextRow + x + 1]
        );

        // Gy: [-1 -1 -1; 0 0 0; 1 1 1]
        const valGy = (
          -1 * src[prevRow + x - 1] - 1 * src[prevRow + x] - 1 * src[prevRow + x + 1] +
           1 * src[nextRow + x - 1] + 1 * src[nextRow + x] + 1 * src[nextRow + x + 1]
        );

        mag[currRow + x] = Math.sqrt(valGx * valGx + valGy * valGy);
      }
    }

    const edges = this.edgeBuffer;
    edges.fill(0);
    for (let i = 0; i < w * h; i++) {
      edges[i] = mag[i] >= threshold ? Math.min(255, mag[i] * 1.5) : 0;
    }
    return edges;
  }

  // Scharr Operator (Optimized gradient calculation for rotational symmetry)
  scharr(options = {}) {
    const w = this.width;
    const h = this.height;
    const src = options.blur ? this.blurBuffer : this.grayBuffer;
    const mag = this.magBuffer;
    const threshold = options.threshold || 60;

    for (let y = 1; y < h - 1; y++) {
      const prevRow = (y - 1) * w;
      const currRow = y * w;
      const nextRow = (y + 1) * w;

      for (let x = 1; x < w - 1; x++) {
        // Gx: [-3 0 3; -10 0 10; -3 0 3] / 16
        const valGx = (
          -3 * src[prevRow + x - 1] + 3 * src[prevRow + x + 1] +
          -10 * src[currRow + x - 1] + 10 * src[currRow + x + 1] +
          -3 * src[nextRow + x - 1] + 3 * src[nextRow + x + 1]
        ) / 16;

        // Gy: [-3 -10 -3; 0 0 0; 3 10 3] / 16
        const valGy = (
          -3 * src[prevRow + x - 1] - 10 * src[prevRow + x] - 3 * src[prevRow + x + 1] +
           3 * src[nextRow + x - 1] + 10 * src[nextRow + x] + 3 * src[nextRow + x + 1]
        ) / 16;

        mag[currRow + x] = Math.sqrt(valGx * valGx + valGy * valGy);
      }
    }

    const edges = this.edgeBuffer;
    edges.fill(0);
    for (let i = 0; i < w * h; i++) {
      edges[i] = mag[i] >= threshold ? Math.min(255, mag[i] * 2.0) : 0;
    }
    return edges;
  }

  // Laplacian / Laplacian of Gaussian (LoG)
  laplacian(options = {}) {
    const w = this.width;
    const h = this.height;
    const src = options.blur ? this.blurBuffer : this.grayBuffer;
    const edges = this.edgeBuffer;
    edges.fill(0);
    const threshold = options.threshold || 30;

    for (let y = 1; y < h - 1; y++) {
      const prevRow = (y - 1) * w;
      const currRow = y * w;
      const nextRow = (y + 1) * w;

      for (let x = 1; x < w - 1; x++) {
        // Discrete Laplacian 3x3 kernel: [0 1 0; 1 -4 1; 0 1 0]
        const val = Math.abs(
          src[prevRow + x] +
          src[currRow + x - 1] - 4 * src[currRow + x] + src[currRow + x + 1] +
          src[nextRow + x]
        );

        edges[currRow + x] = val >= threshold ? Math.min(255, val * 3) : 0;
      }
    }
    return edges;
  }

  // Canny Edge Detection (Gaussian Blur -> Sobel -> Non-Max Suppression -> Hysteresis)
  canny(options = {}) {
    const w = this.width;
    const h = this.height;
    const lowThresh = options.lowThreshold !== undefined ? options.lowThreshold : 25;
    const highThresh = options.highThreshold !== undefined ? options.highThreshold : 70;

    // 1. Ensure Gaussian blur
    this.gaussianBlur();

    // 2. Compute Gradients & Orientation
    const src = this.blurBuffer;
    const gx = this.gxBuffer;
    const gy = this.gyBuffer;
    const mag = this.magBuffer;
    const nms = this.nmsBuffer;
    nms.fill(0);

    for (let y = 1; y < h - 1; y++) {
      const prevRow = (y - 1) * w;
      const currRow = y * w;
      const nextRow = (y + 1) * w;

      for (let x = 1; x < w - 1; x++) {
        const valGx = (
          -1 * src[prevRow + x - 1] + 1 * src[prevRow + x + 1] +
          -2 * src[currRow + x - 1] + 2 * src[currRow + x + 1] +
          -1 * src[nextRow + x - 1] + 1 * src[nextRow + x + 1]
        );
        const valGy = (
          -1 * src[prevRow + x - 1] - 2 * src[prevRow + x] - 1 * src[prevRow + x + 1] +
           1 * src[nextRow + x - 1] + 2 * src[nextRow + x] + 1 * src[nextRow + x + 1]
        );

        const idx = currRow + x;
        gx[idx] = valGx;
        gy[idx] = valGy;
        mag[idx] = Math.sqrt(valGx * valGx + valGy * valGy);
      }
    }

    // 3. Non-Maximum Suppression (Thinning edges to 1 pixel width along gradient normal)
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const idx = row + x;
        const m = mag[idx];
        if (m === 0) continue;

        // Angle in radians
        const angle = Math.atan2(gy[idx], gx[idx]) * (180 / Math.PI);
        const deg = (angle < 0 ? angle + 180 : angle);

        let q = 255;
        let r = 255;

        // 0 degrees (horizontal edge -> vertical gradient)
        if ((deg >= 0 && deg < 22.5) || (deg >= 157.5 && deg <= 180)) {
          q = mag[row + x + 1];
          r = mag[row + x - 1];
        }
        // 45 degrees (diagonal)
        else if (deg >= 22.5 && deg < 67.5) {
          q = mag[(y + 1) * w + x - 1];
          r = mag[(y - 1) * w + x + 1];
        }
        // 90 degrees (vertical edge -> horizontal gradient)
        else if (deg >= 67.5 && deg < 112.5) {
          q = mag[(y + 1) * w + x];
          r = mag[(y - 1) * w + x];
        }
        // 135 degrees (anti-diagonal)
        else if (deg >= 112.5 && deg < 157.5) {
          q = mag[(y - 1) * w + x - 1];
          r = mag[(y + 1) * w + x + 1];
        }

        if (m >= q && m >= r) {
          nms[idx] = m;
        } else {
          nms[idx] = 0;
        }
      }
    }

    // 4. Double Thresholding
    const edges = this.edgeBuffer;
    edges.fill(0);
    const STRONG = 255;
    const WEAK = 50;

    for (let i = 0; i < w * h; i++) {
      const val = nms[i];
      if (val >= highThresh) {
        edges[i] = STRONG;
      } else if (val >= lowThresh) {
        edges[i] = WEAK;
      } else {
        edges[i] = 0;
      }
    }

    // 5. Edge Tracking by Hysteresis (8-connectivity)
    for (let y = 1; y < h - 1; y++) {
      const currRow = y * w;
      for (let x = 1; x < w - 1; x++) {
        const idx = currRow + x;
        if (edges[idx] === WEAK) {
          // Check if any of 8 neighbors is STRONG
          if (
            edges[(y - 1) * w + x - 1] === STRONG ||
            edges[(y - 1) * w + x] === STRONG ||
            edges[(y - 1) * w + x + 1] === STRONG ||
            edges[currRow + x - 1] === STRONG ||
            edges[currRow + x + 1] === STRONG ||
            edges[(y + 1) * w + x - 1] === STRONG ||
            edges[(y + 1) * w + x] === STRONG ||
            edges[(y + 1) * w + x + 1] === STRONG
          ) {
            edges[idx] = STRONG;
          } else {
            edges[idx] = 0;
          }
        }
      }
    }

    return edges;
  }

  // Render edge map to output ImageData with NVIDIA aesthetic styling
  renderOutput(targetImageData, originalImageData, options = {}) {
    const data = targetImageData.data;
    const origData = originalImageData ? originalImageData.data : null;
    const edges = this.edgeBuffer;
    const len = this.width * this.height;

    const color = options.color || 'nvidia-green';
    const invert = options.invert || false;
    const mode = options.renderMode || 'edge-only'; // 'edge-only', 'overlay', 'cyber-hud'

    let rBase = 118, gBase = 185, bBase = 0; // #76B900 NVIDIA Green

    if (color === 'cyber-cyan') {
      rBase = 0; gBase = 229; bBase = 255;
    } else if (color === 'mono') {
      rBase = 255; gBase = 255; bBase = 255;
    } else if (color === 'amber') {
      rBase = 255; gBase = 176; bBase = 0;
    } else if (color === 'neon-red') {
      rBase = 255; gBase = 59; bBase = 48;
    }

    for (let i = 0, j = 0; i < len; i++, j += 4) {
      const edgeVal = edges[i];

      if (mode === 'overlay' && origData) {
        // Blend edge on top of slightly dimmed or grayscale original
        const origR = origData[j] * 0.45;
        const origG = origData[j + 1] * 0.45;
        const origB = origData[j + 2] * 0.45;

        if (edgeVal > 0) {
          const alpha = edgeVal / 255;
          data[j] = Math.min(255, origR + rBase * alpha);
          data[j + 1] = Math.min(255, origG + gBase * alpha);
          data[j + 2] = Math.min(255, origB + bBase * alpha);
        } else {
          data[j] = origR;
          data[j + 1] = origG;
          data[j + 2] = origB;
        }
        data[j + 3] = 255;
      } else {
        // Pure edge map
        if (invert) {
          // White background with dark edges
          const val = 255 - edgeVal;
          data[j] = (edgeVal > 0) ? (rBase * 0.4) : 240;
          data[j + 1] = (edgeVal > 0) ? (gBase * 0.4) : 243;
          data[j + 2] = (edgeVal > 0) ? (bBase * 0.4) : 246;
          data[j + 3] = 255;
        } else {
          // Dark obsidian background with glowing edges
          if (edgeVal > 0) {
            const intensity = edgeVal / 255;
            data[j] = Math.round(rBase * intensity);
            data[j + 1] = Math.round(gBase * intensity);
            data[j + 2] = Math.round(bBase * intensity);
          } else {
            data[j] = 11;
            data[j + 1] = 13;
            data[j + 2] = 16;
          }
          data[j + 3] = 255;
        }
      }
    }
  }
}

window.EdgeDetector = EdgeDetector;


// Application Controller
/**
 * NVIDIA Vision AI Lab - Application Controller
 * Handles image rendering, edge processing loop, split slider, telemetry, and camera feed
 */

(function () {
  // State
  const state = {
    algorithm: 'canny',
    threshold: 45,
    lowThreshold: 25,
    highThreshold: 75,
    blur: true,
    renderMode: 'edge-only', // 'edge-only' or 'overlay'
    invert: false,
    color: 'nvidia-green',
    engine: 'rtx5090',
    preset: 'dashcam',
    isWebcam: false,
    isLiveSim: false,
    sliderPos: 50, // percent
    isDraggingSlider: false
  };

  // Performance telemetry
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let currentFps = 60;
  let currentLatency = 1.2;

  // DOM Elements
  const originalCanvas = document.getElementById('originalCanvas');
  const edgeCanvas = document.getElementById('edgeCanvas');
  const edgeWrapper = document.getElementById('edgeCanvasWrapper');
  const splitHandle = document.getElementById('splitSliderHandle');
  const canvasContainer = document.getElementById('canvasContainer');

  const origCtx = originalCanvas.getContext('2d', { willReadFrequently: true });
  const edgeCtx = edgeCanvas.getContext('2d', { willReadFrequently: true });

  const detector = new EdgeDetector();

  let videoElement = null;
  let animationId = null;

  // Code snippets for developer tabs
  const codeSnippets = {
    cuda: `// NVIDIA CUDA 12.x Optimized Sobel / Canny Edge Detection Kernel
// High throughput with shared memory 2D tile staging
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_W 16
#define TILE_H 16
#define BORDER 1

__global__ void sobel_edge_kernel(
    const unsigned char* __restrict__ d_in,
    unsigned char* __restrict__ d_out,
    int width, int height, int pitch, float threshold) 
{
    __shared__ float s_tile[TILE_H + 2 * BORDER][TILE_W + 2 * BORDER];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Load main tile and halo regions into fast shared memory
    int sm_x = tx + BORDER;
    int sm_y = ty + BORDER;

    if (gx < width && gy < height) {
        s_tile[sm_y][sm_x] = d_in[gy * pitch + gx];
    } else {
        s_tile[sm_y][sm_x] = 0.0f;
    }

    // Top & Bottom halo loads
    if (ty < BORDER) {
        int top_y = max(0, gy - BORDER);
        int bot_y = min(height - 1, gy + TILE_H);
        s_tile[ty][sm_x] = d_in[top_y * pitch + gx];
        s_tile[sm_y + TILE_H][sm_x] = d_in[bot_y * pitch + gx];
    }
    // Left & Right halo loads
    if (tx < BORDER) {
        int left_x = max(0, gx - BORDER);
        int right_x = min(width - 1, gx + TILE_W);
        s_tile[sm_y][tx] = d_in[gy * pitch + left_x];
        s_tile[sm_y][sm_x + TILE_W] = d_in[gy * pitch + right_x];
    }

    __syncthreads();

    if (gx < width && gy < height) {
        // Compute 3x3 Sobel Gradients
        float gx_val = (
            -1.0f * s_tile[sm_y - 1][sm_x - 1] + 1.0f * s_tile[sm_y - 1][sm_x + 1] +
            -2.0f * s_tile[sm_y    ][sm_x - 1] + 2.0f * s_tile[sm_y    ][sm_x + 1] +
            -1.0f * s_tile[sm_y + 1][sm_x - 1] + 1.0f * s_tile[sm_y + 1][sm_x + 1]
        );

        float gy_val = (
            -1.0f * s_tile[sm_y - 1][sm_x - 1] - 2.0f * s_tile[sm_y - 1][sm_x] - 1.0f * s_tile[sm_y - 1][sm_x + 1] +
             1.0f * s_tile[sm_y + 1][sm_x - 1] + 2.0f * s_tile[sm_y + 1][sm_x] + 1.0f * s_tile[sm_y + 1][sm_x + 1]
        );

        float mag = sqrtf(gx_val * gx_val + gy_val * gy_val);
        d_out[gy * pitch + gx] = (mag >= threshold) ? 255 : 0;
    }
}`,

    tensorrt: `# NVIDIA TensorRT 10.x & PyTorch Real-Time Edge Pipeline
import torch
import torch.nn as nn
import tensorrt as trt

class TensorRTEdgeModule(nn.Module):
    """Deep High-Precision Edge Gradient with Conv2d Tensor Cores"""
    def __init__(self):
        super().__init__()
        # Sobel convolution kernels in FP16 for TensorRT engine export
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        self.conv_gx = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_gy = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_gx.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.conv_gy.weight = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x_gray):
        # x_gray: [B, 1, H, W] on CUDA device
        gx = self.conv_gx(x_gray)
        gy = self.conv_gy(x_gray)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return mag

# Export to ONNX and compile with TensorRT trtexec:
# trtexec --onnx=edge_detector.onnx --saveEngine=edge_detector.engine --fp16 --avgRuns=100`,

    opencv: `// NVIDIA Jetson Orin - Hardware Accelerated Edge Inference via GStreamer
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

int main() {
    // Pipeline utilizing Jetson NVMM Hardware ISP
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, "
                           "format=NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink";
                           
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    cv::cuda::GpuMat d_frame, d_gray, d_edges;
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50.0, 100.0);

    while (cap.read(d_frame)) {
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);
        canny->detect(d_gray, d_edges); // 0.8ms on Jetson AGX Orin DLA
    }
    return 0;
}`
  };

  // Hardware Profiles metadata
  const hardwareProfiles = {
    rtx5090: { name: 'NVIDIA RTX 5090 (Blackwell FP16)', baseLatency: 0.7, vram: '1.4 / 32 GB', util: '24%', power: '145W' },
    jetson: { name: 'NVIDIA Jetson AGX Orin 64GB', baseLatency: 2.1, vram: '0.8 / 64 GB', util: '48%', power: '32W' },
    clara: { name: 'NVIDIA Clara Holoscan Medical', baseLatency: 1.1, vram: '2.1 / 48 GB', util: '35%', power: '98W' },
    cpu: { name: 'x86_64 AVX-512 Host CPU', baseLatency: 14.8, vram: '0.4 / 16 GB', util: '88%', power: '65W' }
  };

  // Procedural Sample Scene Generators (Supports real-time dynamic animation with time)
  function generateProceduralScene(preset, w, h, time = 0) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    if (preset === 'dashcam') {
      // Autonomous Highway Dashcam Night View
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, '#040711');
      grad.addColorStop(0.48, '#0b1220');
      grad.addColorStop(0.5, '#1e2638');
      grad.addColorStop(1, '#0c1017');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Horizon & mountain ridges
      ctx.strokeStyle = '#2d3748';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, h * 0.48);
      for (let x = 0; x <= w; x += 30) {
        ctx.lineTo(x, h * 0.48 - Math.sin(x * 0.02) * 25);
      }
      ctx.stroke();

      // Perspective Highway Road
      ctx.fillStyle = '#141a24';
      ctx.beginPath();
      ctx.moveTo(w * 0.48, h * 0.5);
      ctx.lineTo(w * 0.52, h * 0.5);
      ctx.lineTo(w * 0.95, h);
      ctx.lineTo(w * 0.05, h);
      ctx.closePath();
      ctx.fill();

      // Moving Highway lane markings
      const dashOffset = (time * 0.08) % 40;
      ctx.strokeStyle = '#f6ad55';
      ctx.lineWidth = 4;
      ctx.setLineDash([20, 20]);
      ctx.lineDashOffset = -dashOffset;
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.5);
      ctx.lineTo(w * 0.5, h);
      ctx.stroke();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(w * 0.48, h * 0.5);
      ctx.lineTo(w * 0.12, h);
      ctx.moveTo(w * 0.52, h * 0.5);
      ctx.lineTo(w * 0.88, h);
      ctx.stroke();

      // Overhead highway gantry sign
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(w * 0.2, h * 0.18, w * 0.6, h * 0.14);
      ctx.strokeStyle = '#76b900';
      ctx.lineWidth = 3;
      ctx.strokeRect(w * 0.2, h * 0.18, w * 0.6, h * 0.14);

      ctx.fillStyle = '#76b900';
      ctx.font = 'bold 16px monospace';
      ctx.fillText('NVIDIA DRIVE AGX // AUTONOMOUS VISION PILOT', w * 0.24, h * 0.26);

      // Vehicles ahead with movement
      const carOffset = Math.sin(time * 0.003) * 15;
      ctx.fillStyle = '#1f2937';
      ctx.fillRect(w * 0.44 + carOffset, h * 0.58, 70, 40);
      ctx.fillStyle = '#ef4444'; // Red tail lights
      ctx.fillRect(w * 0.45 + carOffset, h * 0.64, 14, 8);
      ctx.fillRect(w * 0.51 + carOffset, h * 0.64, 14, 8);

      // Right lane vehicle
      const car2Offset = Math.cos(time * 0.002) * 10;
      ctx.fillStyle = '#111827';
      ctx.fillRect(w * 0.62 + car2Offset, h * 0.68, 110, 60);
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(w * 0.64 + car2Offset, h * 0.77, 20, 10);
      ctx.fillRect(w * 0.73 + car2Offset, h * 0.77, 20, 10);

      // Streetlamp posts
      for (let i = 0; i < 5; i++) {
        const lx = w * 0.15 - i * 35;
        const ly = h * 0.48 + i * 50;
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(lx, ly - 80);
        ctx.lineTo(lx + 20, ly - 90);
        ctx.stroke();

        ctx.fillStyle = '#fef08a';
        ctx.beginPath();
        ctx.arc(lx + 20, ly - 90, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    } 
    else if (preset === 'robotics') {
      // Jetson Robotics Inspection Arm
      ctx.fillStyle = '#0a0d14';
      ctx.fillRect(0, 0, w, h);

      // Technical calibration grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Dynamic rotation angle
      const armAngle = -Math.PI / 4 + Math.sin(time * 0.002) * 0.2;
      const arm2Angle = Math.PI / 2.2 + Math.cos(time * 0.003) * 0.25;

      // Base joint
      ctx.fillStyle = '#334155';
      ctx.strokeStyle = '#76b900';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(w * 0.25, h * 0.75, 70, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Arm Link 1
      ctx.save();
      ctx.translate(w * 0.25, h * 0.75);
      ctx.rotate(armAngle);
      ctx.fillStyle = '#475569';
      ctx.fillRect(0, -25, 200, 50);
      ctx.strokeRect(0, -25, 200, 50);

      // Joint 2
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.arc(200, 0, 35, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Arm Link 2
      ctx.translate(200, 0);
      ctx.rotate(arm2Angle);
      ctx.fillStyle = '#64748b';
      ctx.fillRect(0, -20, 160, 40);
      ctx.strokeRect(0, -20, 160, 40);

      // Gripper Tool
      ctx.fillStyle = '#76b900';
      ctx.fillRect(160, -30, 40, 15);
      ctx.fillRect(160, 15, 40, 15);
      ctx.restore();

      // Inspection Target PCB
      ctx.fillStyle = '#064e3b';
      ctx.fillRect(w * 0.58, h * 0.35, 240, 180);
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 2;
      ctx.strokeRect(w * 0.58, h * 0.35, 240, 180);

      // Laser scan crossline
      const scanY = h * 0.35 + ((time * 0.06) % 180);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(w * 0.58, scanY);
      ctx.lineTo(w * 0.58 + 240, scanY);
      ctx.stroke();

      // IC Chips & traces
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(w * 0.63, h * 0.42, 70, 70);
      ctx.strokeStyle = '#94a3b8';
      for (let pin = 0; pin < 70; pin += 10) {
        ctx.strokeRect(w * 0.63 + pin, h * 0.40, 4, 4);
        ctx.strokeRect(w * 0.63 + pin, h * 0.53, 4, 4);
      }
    } 
    else if (preset === 'wafer') {
      // Silicon Wafer Inspection
      ctx.fillStyle = '#05070a';
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(w, h) * 0.44;

      const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, r);
      grad.addColorStop(0, '#1e293b');
      grad.addColorStop(0.7, '#0f172a');
      grad.addColorStop(1, '#020617');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#76b900';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Die grid
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r - 5, 0, Math.PI * 2);
      ctx.clip();

      const dieSize = 36;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1;
      for (let x = cx - r; x <= cx + r; x += dieSize) {
        for (let y = cy - r; y <= cy + r; y += dieSize) {
          ctx.strokeRect(x, y, dieSize - 4, dieSize - 4);
          ctx.fillStyle = '#1e1b4b';
          ctx.fillRect(x + 4, y + 4, dieSize - 12, dieSize - 12);
        }
      }
      ctx.restore();

      // Animated inspection laser bar
      const laserY = (cy - r) + ((time * 0.08) % (r * 2));
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - r, laserY);
      ctx.lineTo(cx + r, laserY);
      ctx.stroke();

      // Alignment notch
      ctx.fillStyle = '#05070a';
      ctx.beginPath();
      ctx.arc(cx, cy - r, 16, 0, Math.PI);
      ctx.fill();
    } 
    else if (preset === 'medical') {
      // NVIDIA Clara MRI Neuroimaging Scan
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;

      // Outer Skull Oval
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.ellipse(cx, cy, 180, 230, 0, 0, Math.PI * 2);
      ctx.stroke();

      // Animated brain cortex folds
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 3;
      const breathe = Math.sin(time * 0.002) * 4;
      for (let angle = 0; angle < Math.PI * 2; angle += 0.12) {
        const radX = 140 + breathe + Math.sin(angle * 7) * 22;
        const radY = 185 + breathe + Math.cos(angle * 5) * 24;
        const px = cx + Math.cos(angle) * radX;
        const py = cy + Math.sin(angle) * radY;
        if (angle === 0) ctx.beginPath();
        ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();

      // Ventricles (butterfly shape)
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.ellipse(cx - 30, cy - 10, 25, 60, -0.2, 0, Math.PI * 2);
      ctx.ellipse(cx + 30, cy - 10, 25, 60, 0.2, 0, Math.PI * 2);
      ctx.fill();

      // Scan telemetry crosshairs
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy - 260); ctx.lineTo(cx, cy + 260);
      ctx.moveTo(cx - 240, cy); ctx.lineTo(cx + 240, cy);
      ctx.stroke();
    }

    return canvas;
  }

  // Synchronize CSS dimensions of edgeCanvas to match originalCanvas exactly
  function syncCanvasSizes() {
    if (!originalCanvas || !edgeCanvas) return;
    const rect = originalCanvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      edgeCanvas.style.width = rect.width + 'px';
      edgeCanvas.style.height = rect.height + 'px';
    }
  }

  // Load preset image
  function loadPreset(presetName) {
    state.preset = presetName;
    stopWebcam();
    stopLiveSim();

    const w = 840;
    const h = 540;
    originalCanvas.width = w;
    originalCanvas.height = h;
    edgeCanvas.width = w;
    edgeCanvas.height = h;

    const source = generateProceduralScene(presetName, w, h);
    origCtx.drawImage(source, 0, 0);

    requestAnimationFrame(() => {
      syncCanvasSizes();
      processEdgeDetection();
    });
  }

  // Process edge detection on the loaded original frame
  function processEdgeDetection() {
    const t0 = performance.now();
    const w = originalCanvas.width;
    const h = originalCanvas.height;

    if (!w || !h) return;

    // Resize edge detector buffers if needed
    detector.resizeBuffers(w, h);

    // Get pixel data from original canvas
    const origData = origCtx.getImageData(0, 0, w, h);
    detector.toGrayscale(origData);

    if (state.blur) {
      detector.gaussianBlur();
    }

    // Run selected algorithm
    if (state.algorithm === 'canny') {
      detector.canny({
        lowThreshold: state.lowThreshold,
        highThreshold: state.highThreshold
      });
    } else if (state.algorithm === 'sobel') {
      detector.sobel({
        threshold: state.threshold,
        blur: state.blur,
        binary: false,
        sensitivity: 1.4
      });
    } else if (state.algorithm === 'prewitt') {
      detector.prewitt({
        threshold: state.threshold,
        blur: state.blur
      });
    } else if (state.algorithm === 'scharr') {
      detector.scharr({
        threshold: state.threshold,
        blur: state.blur
      });
    } else if (state.algorithm === 'laplacian') {
      detector.laplacian({
        threshold: state.threshold,
        blur: state.blur
      });
    }

    // Render processed output
    const edgeData = edgeCtx.createImageData(w, h);
    detector.renderOutput(edgeData, origData, {
      color: state.color,
      invert: state.invert,
      renderMode: state.renderMode
    });

    edgeCtx.putImageData(edgeData, 0, 0);

    const t1 = performance.now();
    const actualElapsed = t1 - t0;

    // Calculate Edge Pixel Density
    let edgePixelCount = 0;
    const edges = detector.edgeBuffer;
    const totalPixels = w * h;
    for (let i = 0; i < totalPixels; i++) {
      if (edges[i] > 0) edgePixelCount++;
    }
    const edgeDensity = ((edgePixelCount / totalPixels) * 100).toFixed(1);

    updateTelemetry(actualElapsed, edgeDensity, w, h);
  }

  // Update Telemetry Display
  function updateTelemetry(actualLatency, edgeDensity, w, h) {
    const profile = hardwareProfiles[state.engine] || hardwareProfiles.rtx5090;
    
    // Smooth FPS computation
    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 500) {
      currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
      frameCount = 0;
      lastFrameTime = now;
    }

    // Scaled latency based on chosen hardware simulator
    const reportedLatency = (profile.baseLatency + (actualLatency * 0.15)).toFixed(2);

    const hudRes = document.getElementById('hudRes');
    const hudFps = document.getElementById('hudFps');
    const hudLatency = document.getElementById('hudLatency');
    const hudAlgo = document.getElementById('hudAlgo');
    const hudDensity = document.getElementById('hudDensity');

    if (hudRes) hudRes.textContent = `${w} x ${h}`;
    if (hudFps) hudFps.textContent = `${Math.min(currentFps, 60)} FPS`;
    if (hudLatency) hudLatency.textContent = `${reportedLatency} ms`;
    if (hudAlgo) hudAlgo.textContent = state.algorithm.toUpperCase();
    if (hudDensity) hudDensity.textContent = `${edgeDensity}%`;

    // Hardware Card Telemetry
    const hwLatency = document.getElementById('hwLatency');
    const hwVram = document.getElementById('hwVram');
    const hwTensor = document.getElementById('hwTensor');
    const hwPower = document.getElementById('hwPower');

    if (hwLatency) hwLatency.textContent = `${reportedLatency} ms`;
    if (hwVram) hwVram.textContent = profile.vram;
    if (hwTensor) hwTensor.textContent = profile.util;
    if (hwPower) hwPower.textContent = profile.power;
  }

  // Split Screen Comparison Slider Drag Handling
  function updateSliderPosition(clientX) {
    const rect = canvasContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(rect.width, x));
    const percent = (x / rect.width) * 100;
    state.sliderPos = percent;

    edgeWrapper.style.width = `${percent}%`;
    splitHandle.style.left = `${percent}%`;
  }

  // Webcam Streaming
  async function toggleWebcam() {
    const btn = document.getElementById('webcamBtn');
    if (state.isWebcam) {
      stopWebcam();
      loadPreset(state.preset);
      return;
    }

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Webcam access (getUserMedia) is not supported in this browser. Please check browser permissions.");
      }

      btn.innerHTML = `<span style="display:inline-block;animation:pulse 1s infinite">CONNECTING...</span>`;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: "user"
        },
        audio: false
      });

      if (!videoElement) {
        videoElement = document.createElement('video');
        videoElement.muted = true;
        videoElement.defaultMuted = true;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.setAttribute('muted', '');
        videoElement.setAttribute('autoplay', '');
        videoElement.setAttribute('playsinline', '');
        // Append hidden in body so the browser decodes active frames
        videoElement.style.position = 'fixed';
        videoElement.style.top = '-9999px';
        videoElement.style.left = '-9999px';
        videoElement.style.width = '1px';
        videoElement.style.height = '1px';
        videoElement.style.opacity = '0';
        videoElement.style.pointerEvents = 'none';
        document.body.appendChild(videoElement);
      }

      videoElement.srcObject = stream;

      // Wait until metadata is loaded and video is playing
      await new Promise((resolve) => {
        let isDone = false;
        const markReady = () => {
          if (!isDone) {
            isDone = true;
            resolve();
          }
        };

        videoElement.onloadedmetadata = markReady;
        videoElement.oncanplay = markReady;

        videoElement.play().then(markReady).catch(() => {
          videoElement.muted = true;
          videoElement.play().then(markReady).catch(markReady);
        });

        setTimeout(markReady, 1000);
      });

      const vw = videoElement.videoWidth || 840;
      const vh = videoElement.videoHeight || 540;

      originalCanvas.width = vw;
      originalCanvas.height = vh;
      edgeCanvas.width = vw;
      edgeCanvas.height = vh;

      syncCanvasSizes();

      state.isWebcam = true;
      btn.classList.add('active');
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg> STOP WEBCAM`;

      // Start continuous processing loop
      function frameLoop() {
        if (!state.isWebcam) return;

        if (videoElement && videoElement.readyState >= 2) {
          // If camera aspect ratio changes dynamically, sync canvases
          if (videoElement.videoWidth > 0 && originalCanvas.width !== videoElement.videoWidth) {
            originalCanvas.width = videoElement.videoWidth;
            originalCanvas.height = videoElement.videoHeight;
            edgeCanvas.width = videoElement.videoWidth;
            edgeCanvas.height = videoElement.videoHeight;
            syncCanvasSizes();
          }

          origCtx.drawImage(videoElement, 0, 0, originalCanvas.width, originalCanvas.height);
          processEdgeDetection();
        }

        animationId = requestAnimationFrame(frameLoop);
      }
      frameLoop();

    } catch (err) {
      console.warn("Webcam access error:", err);
      alert("Camera could not be started: " + (err.message || "Please grant camera permission in your browser URL bar."));
      stopWebcam();
      loadPreset(state.preset);
    }
  }

  function stopWebcam() {
    state.isWebcam = false;
    const btn = document.getElementById('webcamBtn');
    if (btn) {
      btn.classList.remove('active');
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg> LIVE WEBCAM`;
    }
    if (videoElement && videoElement.srcObject) {
      videoElement.srcObject.getTracks().forEach(track => track.stop());
      videoElement.srcObject = null;
    }
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // Image Upload handler
  function handleImageUpload(file) {
    if (!file || !file.type.startsWith('image/')) return;
    stopWebcam();
    stopLiveSim();

    const reader = new FileReader();
    reader.onload = function (e) {
      const img = new Image();
      img.onload = function () {
        // Limit max dimension for responsive performance
        const maxDim = 1280;
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }

        originalCanvas.width = w;
        originalCanvas.height = h;
        edgeCanvas.width = w;
        edgeCanvas.height = h;

        origCtx.drawImage(img, 0, 0, w, h);
        requestAnimationFrame(() => {
          syncCanvasSizes();
          processEdgeDetection();
        });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // Live Simulation Feed (Real-Time 60 FPS Procedural Stream)
  function toggleLiveSim() {
    const btn = document.getElementById('liveSimBtn');
    if (state.isLiveSim) {
      stopLiveSim();
      loadPreset(state.preset);
      return;
    }
    stopWebcam();
    state.isLiveSim = true;
    btn.classList.add('active');
    btn.style.background = 'rgba(0, 229, 255, 0.2)';
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg> STOP SIM`;

    const w = 840;
    const h = 540;
    originalCanvas.width = w;
    originalCanvas.height = h;
    edgeCanvas.width = w;
    edgeCanvas.height = h;
    syncCanvasSizes();

    function simLoop(timestamp) {
      if (!state.isLiveSim) return;
      const source = generateProceduralScene(state.preset, w, h, timestamp);
      origCtx.drawImage(source, 0, 0);
      processEdgeDetection();
      animationId = requestAnimationFrame(simLoop);
    }
    animationId = requestAnimationFrame(simLoop);
  }

  function stopLiveSim() {
    state.isLiveSim = false;
    const btn = document.getElementById('liveSimBtn');
    if (btn) {
      btn.classList.remove('active');
      btn.style.background = '';
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> LIVE SIM (60FPS)`;
    }
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  // Export processed canvas to PNG
  function exportEdgeImage() {
    const link = document.createElement('a');
    link.download = `nvidia-edge-lab-${state.algorithm}-${Date.now()}.png`;
    link.href = edgeCanvas.toDataURL('image/png');
    link.click();
  }

  // Initialize UI Bindings
  function initListeners() {
    // Split Slider Drag events
    splitHandle.addEventListener('mousedown', () => { state.isDraggingSlider = true; });
    window.addEventListener('mouseup', () => { state.isDraggingSlider = false; });
    window.addEventListener('mousemove', (e) => {
      if (state.isDraggingSlider) {
        updateSliderPosition(e.clientX);
      }
    });

    // Touch events for mobile/tablet
    splitHandle.addEventListener('touchstart', () => { state.isDraggingSlider = true; }, { passive: true });
    window.addEventListener('touchend', () => { state.isDraggingSlider = false; });
    window.addEventListener('touchmove', (e) => {
      if (state.isDraggingSlider && e.touches[0]) {
        updateSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    // Preset Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadPreset(btn.dataset.preset);
      });
    });

    // Algorithm Tabs
    document.querySelectorAll('.algo-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.algo-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.algorithm = tab.dataset.algo;

        // Toggle visibility of Canny-specific sliders
        const cannyControls = document.getElementById('cannyControls');
        const standardControls = document.getElementById('standardControls');
        if (state.algorithm === 'canny') {
          cannyControls.style.display = 'flex';
          standardControls.style.display = 'none';
        } else {
          cannyControls.style.display = 'none';
          standardControls.style.display = 'flex';
        }

        processEdgeDetection();
      });
    });

    // Sliders
    const threshSlider = document.getElementById('threshSlider');
    const threshVal = document.getElementById('threshVal');
    threshSlider.addEventListener('input', (e) => {
      state.threshold = parseInt(e.target.value, 10);
      threshVal.textContent = state.threshold;
      processEdgeDetection();
    });

    const cannyLowSlider = document.getElementById('cannyLowSlider');
    const cannyLowVal = document.getElementById('cannyLowVal');
    cannyLowSlider.addEventListener('input', (e) => {
      state.lowThreshold = parseInt(e.target.value, 10);
      cannyLowVal.textContent = state.lowThreshold;
      processEdgeDetection();
    });

    const cannyHighSlider = document.getElementById('cannyHighSlider');
    const cannyHighVal = document.getElementById('cannyHighVal');
    cannyHighSlider.addEventListener('input', (e) => {
      state.highThreshold = parseInt(e.target.value, 10);
      cannyHighVal.textContent = state.highThreshold;
      processEdgeDetection();
    });

    // Toggles
    const blurToggle = document.getElementById('blurToggle');
    blurToggle.addEventListener('change', (e) => {
      state.blur = e.target.checked;
      processEdgeDetection();
    });

    const invertToggle = document.getElementById('invertToggle');
    invertToggle.addEventListener('change', (e) => {
      state.invert = e.target.checked;
      processEdgeDetection();
    });

    const overlayToggle = document.getElementById('overlayToggle');
    overlayToggle.addEventListener('change', (e) => {
      state.renderMode = e.target.checked ? 'overlay' : 'edge-only';
      processEdgeDetection();
    });

    // Color Palette Chips
    document.querySelectorAll('.palette-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.palette-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        state.color = chip.dataset.color;
        processEdgeDetection();
      });
    });

    // Hardware Engine Selector
    const engineSelect = document.getElementById('engineSelect');
    engineSelect.addEventListener('change', (e) => {
      state.engine = e.target.value;
      const profile = hardwareProfiles[state.engine];
      document.getElementById('navHwName').textContent = profile.name.split('(')[0].trim();
      processEdgeDetection();
    });

    // Webcam Button
    const webcamBtn = document.getElementById('webcamBtn');
    webcamBtn.addEventListener('click', toggleWebcam);

    // Live Sim Button
    const liveSimBtn = document.getElementById('liveSimBtn');
    if (liveSimBtn) {
      liveSimBtn.addEventListener('click', toggleLiveSim);
    }

    // Image Upload
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageUpload(e.target.files[0]);
      }
    });

    // Export Button
    document.getElementById('exportBtn').addEventListener('click', exportEdgeImage);

    // Research Code Tabs
    const codePre = document.getElementById('codeDisplay');
    const copyBtn = document.getElementById('copyCodeBtn');

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tabKey = btn.dataset.tab;
        if (codeSnippets[tabKey]) {
          codePre.textContent = codeSnippets[tabKey];
          document.getElementById('codeTabContent').style.display = 'block';
          document.getElementById('benchTabContent').style.display = 'none';
        } else if (tabKey === 'benchmarks') {
          document.getElementById('codeTabContent').style.display = 'none';
          document.getElementById('benchTabContent').style.display = 'block';
        }
      });
    });

    // Copy Code Button
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(codePre.textContent).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'COPIED!';
        copyBtn.style.borderColor = 'var(--nvidia-green)';
        copyBtn.style.color = 'var(--nvidia-green)';
        setTimeout(() => {
          copyBtn.textContent = originalText;
          copyBtn.style.borderColor = '';
          copyBtn.style.color = '';
        }, 1800);
      });
    });

    // Window Resize Sync
    window.addEventListener('resize', syncCanvasSizes);

    // Set initial code content
    codePre.textContent = codeSnippets.cuda;
  }

  // Start Application
  window.addEventListener('DOMContentLoaded', () => {
    initListeners();
    loadPreset('dashcam');
  });
})();

