/**
 * This file is part of REVO.
 *
 * Copyright (C) 2014-2017 Schenk Fabian <schenk at icg dot tugraz dot at> (Graz
 * University of Technology) For more information see
 * <https://github.com/fabianschenk/REVO/>
 *
 * REVO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * REVO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with REVO. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once
#include "../datastructures/imgpyramidrgbd.h"
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>

#ifdef WITH_REALSENSE
#include "realsensesensor.h"
#endif

class IOWrapperSettings {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IOWrapperSettings(const std::string &settingsFile, int nRuns = 0);
  bool inline READ_FROM_DATASET() const {
    return !(READ_FROM_ASTRA_PRO || READ_FROM_REALSENSE);
  }
  bool READ_INTRINSICS_FROM_SENSOR;
  bool READ_FROM_ASTRA_PRO;
  bool READ_FROM_REALSENSE;
  bool READ_FROM_ASTRA;
  int SKIP_FIRST_N_FRAMES;
  int READ_N_IMAGES;
  float DEPTH_SCALE_FACTOR;
  bool useDepthTimeStamp;
  // bool DO_ADAPT_CANNY_VALUES;
  bool DO_WAIT_AUTOEXP;
  std::vector<std::string> datasets;
  cv::Size2i imgSize;
  std::string MainFolder;
  std::string subDataset;
  std::string associateFile;
  std::string poseOutDir="./result";
  bool DO_OUTPUT_IMAGES;
  bool READ_FROM_ASTRA_DATA;
  bool isFinished;
};

class IOWrapperRGBD {
private:
  //    FileManager mFileReader;
  std::ifstream fileList;
  std::ofstream associateFile;
  IOWrapperSettings mSettings;
  bool mQuitFlag;
  cv::Mat rgb, depth;
  int nFrames;
  // std::queue<ImgPyramidRGBD> mPyrQueue;
  std::queue<std::unique_ptr<ImgPyramidRGBD>> mPyrQueue;
  std::mutex mtx;
  bool mFinish;
  bool mInitSuccess;
  bool mAllImagesRead;

  ImgPyramidSettings mPyrConfig;
  std::shared_ptr<CameraPyr> mCamPyr;
  std::string outputImgDir;
  bool mHasMoreImages;
  void requestQuit();

  // different modalities
  void generateImgPyramidFromFiles();
#ifdef WITH_REALSENSE
  std::unique_ptr<RealsenseSensor> realSenseSensor;
#endif
  // void adaptCannyValues();
  void generateImgPyramidFromAstraPro();
  void generateImgPyramidFromAstra();
  void generateImgPyramidFromRealSense();
  void writeImages(const cv::Mat &rgb, const cv::Mat depth,
                   const float timestamp);
  bool readNextFrame(cv::Mat &rgb, cv::Mat &depth, double &rgbTimeStamp,
                     double &depthTimeStamp, int skipFrames,
                     double depthScaleFactor);
  int noFrames;

public:
  void generateImgPyramid() {
    if (mSettings.READ_FROM_ASTRA_PRO)
      generateImgPyramidFromAstraPro();
    else if (mSettings.READ_FROM_REALSENSE)
      generateImgPyramidFromRealSense();
    else if (mSettings.READ_FROM_ASTRA) {
      generateImgPyramidFromAstra();
    } else
      generateImgPyramidFromFiles();
  }
  IOWrapperRGBD(const IOWrapperSettings &settings,
                const ImgPyramidSettings &mPyrSettings,
                const std::shared_ptr<CameraPyr> &camPyr);
  //~IOWrapperRGBD();
  inline bool isImgPyramidAvailable() {
    // std::unique_lock<std::mutex> lock(this->mtx);
    I3D_LOG(i3d::detail) << "isImgPyramidAvailable";
    return mPyrQueue.size() > 0;
  }
  inline bool hasMoreImages() { return mHasMoreImages; }
  bool getOldestPyramid(ImgPyramidRGBD &pyr);
  bool getOldestPyramid(std::shared_ptr<ImgPyramidRGBD> &pyr);
  void setFinish(bool setFinish);
  inline bool isInitSuccess() {
    if (mSettings.isFinished) {
      return false;
    }
    return mInitSuccess;
  }
};
