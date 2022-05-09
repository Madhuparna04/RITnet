#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "common/switchboard.hpp"
#include "common/threadloop.hpp"
#include "common/data_format.hpp"

namespace fs = std::filesystem;
using namespace ILLIXR;

std::string path = "/home/madhuparna/ILLIXR_Copy/RITnet/Semantic_Segmentation_Dataset/test/images";
fs::directory_iterator end_itr;
fs::directory_iterator entry( path );

int read_images = 0;
#define WAITKEY (1)

class eye_tracking : public plugin {
public:
    eye_tracking(std::string name_, phonebook *pb_)
        : plugin{name_, pb_}
        , sb{pb->lookup_impl<switchboard>()}
        , eye_segmentation_{sb->get_writer<eye_segmentation>("eye_segmentation")}
        , gamma_lut_(1, GAMMA_LUT_SIZE, CV_8U)
    {
        try {
            ritnet_ = torch::jit::load("/home/madhuparna/ILLIXR_Copy/RITnet/ritnet.pt");
            ritnet_.eval();
            ritnet_.to(device_);
        } catch (const c10::Error &e) {
            std::cerr << "Could not load RITnet model from disk\n";
            std::cerr << e.msg() << std::endl;
            ILLIXR::abort();
        }

        // Generate CLAHE object
        clahe_ = cv::createCLAHE(CLIP_LIMIT, {TILE_SIZE, TILE_SIZE});

        // Generate gamma correction lookup table
        uchar *ptr = gamma_lut_.ptr();
        for (unsigned i = 0; i < GAMMA_LUT_SIZE; i++)
            ptr[i] = cv::saturate_cast<uchar>(std::pow((float) i / 255.0f, GAMMA) * 255.0f);

        // Subscribe to "imu_cam"
        // TODO: This is just using "imu_cam" for debugging purposes. In reality, this should
        // subscribe to "eye_cam" (or similar), but we don't have an eye camera yet.
        sb->schedule<imu_cam_type>(id, "imu_cam", [&](switchboard::ptr<const imu_cam_type> datum, size_t) {
            callback(datum);
        });
    }

    void callback(switchboard::ptr<const imu_cam_type> datum) {
        // Stop if IMU packet
        if(!datum->img0.has_value() && !datum->img1.has_value()) {
            return;
        }
        

#ifndef NDEBUG
        auto preprocess_start = std::chrono::high_resolution_clock::now();
#endif

        // Extract images
        cv::Mat img0{datum->img0.value()};
        cv::Mat img1{datum->img1.value()};
	
        //Reading images from a folder and performimg segmentation, ellipse fitting and gaze tracking on it
	 if (entry != end_itr) {
	         img0 = cv::imread((cv::String)entry->path(),  cv::IMREAD_GRAYSCALE);
	         entry++;
	 }
        //Reading consecutive two images. Ideally needs to be left and right eye.
	 if (entry != end_itr) {
		 img1 = cv::imread((cv::String)entry->path(),  cv::IMREAD_GRAYSCALE);
		 entry++;
	 }

#ifndef NDEBUG
        cv::imshow("img0 Original", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Original", img1);
        cv::waitKey(WAITKEY);
#endif

        // Convert to grayscale (Needed only if images are colores, currently all images read are already grayscale)
        /*
        cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
        */

#ifndef NDEBUG
        cv::imshow("img0 Grayscale", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Grayscale", img1);
        cv::waitKey(WAITKEY);
#endif
        // Transpose
        cv::transpose(img0, img0);
        cv::transpose(img1, img1);

#ifndef NDEBUG
        cv::imshow("img0 Transposed", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Transposed", img1);
        cv::waitKey(WAITKEY);
#endif

        // Resize to 400x640 for RITnet
        cv::resize(img0, img0, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        cv::resize(img1, img1, cv::Size(IMG_WIDTH, IMG_HEIGHT));

#ifndef NDEBUG
        cv::imshow("img0 Resized", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Resized", img1);
        cv::waitKey(WAITKEY);
#endif

        // Apply gamma correction
        cv::LUT(img0, gamma_lut_, img0);
        cv::LUT(img1, gamma_lut_, img1);

#ifndef NDEBUG
        cv::imshow("img0 Gamma", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Gamma", img1);
        cv::waitKey(WAITKEY);
#endif

        // Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe_->apply(img0, img0);
        clahe_->apply(img1, img1);

#ifndef NDEBUG
        cv::imshow("img0 CLAHE", img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 CLAHE", img1);
        cv::waitKey(WAITKEY);
#endif

#ifndef NDEBUG
        auto preprocess_stop = std::chrono::high_resolution_clock::now();
        std::cout << "Preprocessing time: " << std::chrono::duration_cast<std::chrono::microseconds>(preprocess_stop - preprocess_start).count() << " us\n";
#endif
        // Perform segmentation
        segment(img0, img1);
        return;
        
    };

    // Segments the two input images into pupil, iris, sclera, and none-of-the-above, and
    // publishes the segmented images on the "eye_segmentation" topic on switchboard. The
    // images must be 400x640 in CV_8UC1 format.
    void segment(cv::Mat &img0, cv::Mat &img1) {

    	cv::imwrite( "results_ritnet/" + std::to_string(read_images) + "_before_segment.png", img0 );
    	cv::imwrite( "results_ritnet/" + std::to_string(read_images+1) + "_before_segment.png", img1 );
        // Convert images to tensors and batch them together
        auto tensor0 = mat_to_tensor(img0);
        auto tensor1 = mat_to_tensor(img1);
        auto input = torch::cat({tensor0, tensor1}, 0);
        verify_tensor(input, 1);

        // Transfer input tensor to GPU memory
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.to(device_));

#ifndef NDEBUG
        auto segment_start = std::chrono::high_resolution_clock::now();
#endif

        // Run batched inference on both tensors and copy result back to CPU memory
        auto output = ritnet_.forward(inputs).toTensor().to(torch::kCPU);

#ifndef NDEBUG
        auto segment_stop = std::chrono::high_resolution_clock::now();
        std::cout << "Segmentation time: " << std::chrono::duration_cast<std::chrono::microseconds>(segment_stop - segment_start).count() << " us\n";
        
#endif

        // Make sure output shape looks good: NxCxHxW == 2x4x640x400
        verify_tensor(output, 4);

        // Get max values (predicted classes) and their indices (values, indices)
        auto max = output.max(1);

        // Convert indices into N x H x W predictions tensor
        auto sizes = output.sizes();
        auto predictions = std::get<1>(max).view({sizes[0], sizes[2], sizes[3]});

        // Map predictions to pixel values in [0, 1]
        auto pixels = predictions.div(3.0f);

        // Drop first channel in each tensor to get 640x400 segmented tensor
        auto seg_tensor0 = pixels.narrow(0, 0, 1).squeeze();
        auto seg_tensor1 = pixels.narrow(0, 1, 1).squeeze();

        // Convert tensors to segmented images
        auto rows = static_cast<int>(seg_tensor0.size(0));
        auto cols = static_cast<int>(seg_tensor0.size(1));
        cv::Mat seg_img0 = cv::Mat{rows, cols, CV_32FC1, seg_tensor0.data_ptr<float>()};
        rows = static_cast<int>(seg_tensor1.size(0));
        cols = static_cast<int>(seg_tensor1.size(1));
        cv::Mat seg_img1 = cv::Mat{rows, cols, CV_32FC1, seg_tensor1.data_ptr<float>()};

#ifndef NDEBUG
        cv::imshow("img0 Segmented", seg_img0);
        cv::waitKey(WAITKEY);
        cv::imshow("img1 Segmented", seg_img1);
        cv::waitKey(WAITKEY);
#endif
        // Publish images
        cv::imwrite( "results_ritnet/" + std::to_string(read_images) + "_after_segment.png", 255*seg_img0 );
    	cv::imwrite( "results_ritnet/" + std::to_string(read_images+1) + "_after_segment.png", 255*seg_img1 );


	 //Ideally should be a new plugin that read the segmented image
        ellipse_fit(seg_img0);
        read_images += 1;
        ellipse_fit(seg_img1);
        read_images += 1;
        
        eye_segmentation_.put(eye_segmentation_.allocate<eye_segmentation>(
	   	eye_segmentation{
           	new cv::Mat{seg_img0},
           	new cv::Mat{seg_img1}
        	}
	));
       
    }

void ellipse_fit(cv::Mat img) {
	cv::Mat src = img;
        cv::Mat src_gray;
        
	src.convertTo(src_gray, CV_8UC1);
        int thresh = 100;

        src_gray = 255*src_gray;
        cv::blur( src_gray, src_gray, cv::Size(3,3) );  
        
        auto start = std::chrono::high_resolution_clock::now();
  
        cv::Mat canny_output;
        cv::Canny( src_gray, canny_output, thresh, thresh*2 );
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
        float mini_area = 0;
        int final_index = 0;
        cv::RotatedRect final_box;
        std::cout<<"Contours found"<< contours.size() <<std::endl;

        for( size_t i = 0; i< contours.size(); i++ )
        {
        	if (contours[i].size() < 5)
        		continue;
                cv::Mat pointsf;
                cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
                cv::RotatedRect box = cv::fitEllipse(pointsf);
                //Removing disproportionate i.e. extremely long contours
                if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
                        continue;
                else {
                        if (box.size.height * box.size.width > mini_area) {
                                mini_area = box.size.height * box.size.width;
                                final_index = i;
                                final_box = box;
                        }
                }
        }
        if (mini_area == 0) {
        	std::cout<<" no valid contours"<<std::endl;
        	return;
        }
        cv::RNG rng(12345);
        cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        cv::drawContours( drawing, contours, (int)final_index, color, 2, cv::LINE_8, hierarchy, 0 );
        cv::ellipse(drawing, final_box.center, final_box.size*0.5f, final_box.angle, 0, 360, cv::Scalar(0,255,255), 1, cv::LINE_AA);
        cv::Point2f vtx[4];
        final_box.points(vtx);
        for( int j = 0; j < 4; j++ )
                cv::line(drawing, vtx[j], vtx[(j+1)%4], cv::Scalar(0,255,0), 1, cv::LINE_AA);

        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Ellipse Fitting time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us\n";

    	cv::imwrite( "results_ritnet/" + std::to_string(read_images) + "_ellipse.png", drawing );
        float foveaX = round((final_box.center.x - 200) / 2.42 * (2.42 + 5) + 1080);
        float foveaY = round((final_box.center.y - 320) / 2.42 * (2.42 + 5) + 600);
        std::cout<< "Image number "<< read_images << " foveaX " << foveaX << " foveaY " << foveaY << std::endl;
    }

private:
    // This is not a generic function. It is specialized to the requirements of
    // this plugin. It assumes the input is a grayscale 400x640 image in
    // CV_8UC1 format, and returns a 1x1x640x400 tensor of floats in the range
    // [-1, 1].
    // Using img.ptr() instead of img.data eliminates a copy.
    static inline torch::Tensor mat_to_tensor(cv::Mat &img) {
        // Make sure input looks good: 400x640 grayscale in CV_8UC1 format
        verify_image(img);

        // Read image in (H, W, C) order
        std::vector<int64_t> dims = {1, img.rows, img.cols, img.channels()};
        auto tensor = torch::from_blob(img.ptr<uchar>(), dims, torch::kByte);

        // Map to [-1, 1]
        tensor = tensor.to(torch::kFloat).sub(127.5f).mul(0.007843137f);

        // Convert to (N, C, H, W)
        tensor = tensor.permute({0, 3, 1, 2});
        return tensor;
    }

    // This is not a generic function. It is specialized to the requirements of
    // this plugin. It assumes the input is a 640x400 tensor of floats, and
    // returns a 400x640 image in CV_32FC1 format.
    // Using tensor.data_ptr() instead of tensor.data() eliminates a copy.
    static inline cv::Mat tensor_to_mat(torch::Tensor &tensor) {
        auto rows = static_cast<int>(tensor.size(0));
        auto cols = static_cast<int>(tensor.size(1));
        return cv::Mat{rows, cols, CV_32FC1, tensor.data_ptr<float>()};
    }

    // Verifies image dimensions and data type are correct
    static inline void verify_image(const cv::Mat &img) {
        assert(img.cols == IMG_WIDTH
            && img.rows == IMG_HEIGHT
            && img.channels() == IMG_CHANNELS
            && img.elemSize() == IMG_ELEM_SIZE);
    }

    // Verifies tensor dimensions are correct
    static inline void verify_tensor(const torch::Tensor &tensor, const unsigned channels) {
        assert(tensor.size(0) == NUM_IMAGES
            && tensor.size(1) == channels
            && tensor.size(2) == IMG_HEIGHT
            && tensor.size(3) == IMG_WIDTH);
    }

    // Helper function for printing tensor dimensions
    static inline void print_tensor_dims(const std::string &name, const torch::Tensor &x) {
        std::cout << name << ": [";
        for (const auto size : x.sizes())
            std::cout << size << ",";
        std::cout << "]\n";
    }

private:
    // ILLIXR
    const std::shared_ptr<switchboard> sb;
    switchboard::writer<eye_segmentation> eye_segmentation_;

    // Torch
    torch::jit::script::Module ritnet_;
    torch::Device device_ = torch::kCUDA;

    // CV
    cv::Mat gamma_lut_;
    cv::Ptr<cv::CLAHE> clahe_;

    // Image constants
    static constexpr unsigned NUM_IMAGES    = 2;
    static constexpr unsigned IMG_WIDTH     = 400;
    static constexpr unsigned IMG_HEIGHT    = 640;
    static constexpr unsigned IMG_CHANNELS  = 1;
    static constexpr unsigned IMG_ELEM_SIZE = 1;

    // Image processing constants
    static constexpr float GAMMA            = 0.8;
    static constexpr int GAMMA_LUT_SIZE     = 256;
    static constexpr float CLIP_LIMIT       = 1.5f;
    static constexpr int TILE_SIZE          = 8;
};

PLUGIN_MAIN(eye_tracking);
