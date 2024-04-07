#include "esp_camera.h"
#include <opencv2/opencv.hpp>

#define CASCADE_FILE "haarcascade_frontalface_default.xml"
void setup()
{
    camera_config_t config;
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }
}

void loop()
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
        Serial.println("Camera capture failed");
        return;
    }

    cv::Mat frame(fb->height, fb->width, CV_8UC3, fb->buf);

    cv::CascadeClassifier cascade;
    if (!cascade.load(CASCADE_FILE))
    {
        Serial.println("Error loading cascade classifier");
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> objects;
    cascade.detectMultiScale(gray, objects, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (size_t i = 0; i < objects.size(); ++i)
    {
        cv::rectangle(frame, objects[i], cv::Scalar(255, 0, 0), 2);
    }

}

void app_main()
{
    setup();
    while (true)
    {
        loop();
    }
}