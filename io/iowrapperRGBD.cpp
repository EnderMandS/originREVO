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
#include "iowrapperRGBD.h"
#include "../utils/timer.h"
#include <boost/filesystem.hpp>
#include <unistd.h>

IOWrapperSettings::IOWrapperSettings(const std::string &settingsFile,
                                     int nRuns) {
  isFinished = false;
  cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    I3D_LOG(i3d::error) << "Couldn't open settings file at location: "
                        << settingsFile;
    exit(EXIT_FAILURE);
  }
  // image pyramid settings (camera matrix, resolutions,...)
  int inputType; // 0 = dataset, 1 = ASTRA PRO, 2 = REAL SENSE
  cv::read(fs["INPUT_TYPE"], inputType, 0);
  READ_FROM_ASTRA_PRO = READ_FROM_REALSENSE = READ_FROM_ASTRA = false;
  switch (inputType) {
  case 1:
    READ_FROM_ASTRA_PRO = true;
    if (nRuns > 0)
      isFinished = true;
    break;
  case 2:
    READ_FROM_REALSENSE = true;
    if (nRuns > 0)
      isFinished = true;
    break;
  case 3:
    READ_FROM_ASTRA = true;
    if (nRuns > 0)
      isFinished = true;
    break;
  // dataset
  case 0:
  default:
    // now read all the settings
    // general settings
    // datasets or sensor?
    cv::FileNode n = fs["Datasets"]; // Read string sequence - Get node
    if (n.type() == cv::FileNode::SEQ) {
      cv::FileNodeIterator it = n.begin(),
                           it_end = n.end(); // Go through the node
      for (; it != it_end; ++it)
        datasets.push_back((std::string)*it);
    } else {
      I3D_LOG(i3d::debug) << "Running single dataset. Trying to read string!";
      datasets.push_back(fs["Datasets"]);
    }
    I3D_LOG(i3d::info) << "Datasets: " << datasets.size()
                       << " nRuns: " << nRuns;
    if (nRuns >= int(datasets.size())) {
      I3D_LOG(i3d::info) << "Finished all datasets!";
      isFinished = true;
      subDataset = datasets.back();
    } else
      subDataset = datasets[nRuns];
    break;
  }
  // Ensure that it stops!
  if ((READ_FROM_ASTRA_DATA || READ_FROM_REALSENSE) && nRuns > 0)
    isFinished = true;
  cv::read(fs["READ_INTRINSICS_FROM_SENSOR"], READ_INTRINSICS_FROM_SENSOR,
           false);
  // only needed to skip auto exposure artifacts
  cv::read(fs["SKIP_FIRST_N_FRAMES"], SKIP_FIRST_N_FRAMES, 0);
  cv::read(fs["READ_N_IMAGES"], READ_N_IMAGES,
           100000); // read at least 100000 images
  // cv::read(fs["DO_ADAPT_CANNY_VALUES"],DO_ADAPT_CANNY_VALUES, true);
  // //tries to guess the Canny values from the first frame!
  cv::read(fs["DO_WAIT_AUTOEXP"], DO_WAIT_AUTOEXP,
           false); // skips the first 20 frames or so to avoid auto exposure
                   // problems
  if (SKIP_FIRST_N_FRAMES < 20 && DO_WAIT_AUTOEXP) {
    SKIP_FIRST_N_FRAMES = 20;
    I3D_LOG(i3d::warning)
        << "Skipping first 20 frames to avoid auto exposure problems!";
  }
  if (!READ_INTRINSICS_FROM_SENSOR) {
    cv::read(fs["DEPTH_SCALE_FACTOR"], DEPTH_SCALE_FACTOR, 1000.0f);
    cv::read(fs["Camera.width"], imgSize.width, 640);
    cv::read(fs["Camera.height"], imgSize.height, 480);
  }
  cv::read(fs["DO_RECORD_IMAGES"], DO_OUTPUT_IMAGES, false);
  associateFile = (std::string)fs["ASSOCIATE"];
  if (associateFile.compare(""))
    associateFile = "associate.txt";
  cv::read(fs["useDepthTimeStamp"], useDepthTimeStamp, true);
  MainFolder = (std::string)fs["MainFolder"];
  poseOutDir = (std::string)fs["poseOutDir"];
  I3D_LOG(i3d::info) << "MainFolder: " << MainFolder;
  I3D_LOG(i3d::info) << "poseOutDir: " << poseOutDir;
  fs.release();
}

IOWrapperRGBD::IOWrapperRGBD(const IOWrapperSettings &settings,
                             const ImgPyramidSettings &mPyrSettings,
                             const std::shared_ptr<CameraPyr> &camPyr)
    : mSettings(settings), mQuitFlag(false), rgb(settings.imgSize, CV_8UC3),
      depth(settings.imgSize, CV_32FC1), nFrames(0), mFinish(false),
      mAllImagesRead(false), mPyrConfig(mPyrSettings), mCamPyr(camPyr),
      mHasMoreImages(true), noFrames(0) {
  mInitSuccess = true;
  I3D_LOG(i3d::info) << "camPyr->size(): " << camPyr->size();
  if (mSettings.READ_FROM_DATASET()) {
    // OPEN FILES
    fileList.open((mSettings.MainFolder + "/" + mSettings.subDataset + "/" +
                   mSettings.associateFile)
                      .c_str(),
                  std::ios_base::in);
    I3D_LOG(i3d::info) << "Reading: "
                       << (mSettings.MainFolder + "/" + mSettings.subDataset +
                           "/" + mSettings.associateFile);
    if (!fileList.is_open()) {
      I3D_LOG(i3d::error) << "Could not open associateFile at: "
                          << (mSettings.MainFolder + "/" +
                              mSettings.subDataset + "/" +
                              mSettings.associateFile);
      mQuitFlag = true;
      exit(EXIT_FAILURE);
    }
    assert(fileList.is_open() && "File could not been opened!");
  }
  rgb = cv::Mat(mSettings.imgSize.width, mSettings.imgSize.height, CV_8UC3);
  depth = cv::Mat(mSettings.imgSize.width, mSettings.imgSize.height, CV_32FC1);
  // if (!mFileReader.isFileOpen()) mQuitFlag = true;
}

void IOWrapperRGBD::generateImgPyramidFromAstraPro() {
#ifdef WITH_ORBBEC_ASTRA_PRO
  while (this->mSettings.READ_FROM_ASTRA_PRO && !mFinish) {
    if (this->orbbecAstraProSensor->getImages(rgb, depth,
                                              mSettings.DEPTH_SCALE_FACTOR)) {
      auto start = Timer::getTime();
      nFrames++;
      I3D_LOG(i3d::info) << "nFrames: " << nFrames;
      // there is a strange orbbec bug, where the first two lines of the "color"
      // image are invalid
      rgb.row(2).copyTo(rgb.row(0));
      rgb.row(2).copyTo(rgb.row(1));
      if (mSettings.DO_OUTPUT_IMAGES)
        writeImages(rgb, depth, nFrames);
      I3D_LOG(i3d::info) << mSettings.DEPTH_SCALE_FACTOR;
      // mPyrQueue.push(pyr);
      std::unique_ptr<ImgPyramidRGBD> ptrTmp(std::unique_ptr<ImgPyramidRGBD>(
          new ImgPyramidRGBD(mPyrConfig, mCamPyr, rgb, depth, nFrames)));
      {
        std::unique_lock<std::mutex> lock(this->mtx);
        mPyrQueue.push(std::move(ptrTmp));
        I3D_LOG(i3d::error) << "ImgPyramid queued" << mPyrQueue.size();
      }
      auto end = Timer::getTime();
      I3D_LOG(i3d::info) << "Reading image: " << nFrames << " in "
                         << Timer::getTimeDiffMiS(start, end);
    }
    usleep(500);
  }
#else
  I3D_LOG(i3d::error) << "Not compiled with Orbbec Astra support!";
  exit(0);
#endif
}
// void IOWrapperRGBD::adaptCannyValues()
//{
//    if (flagAdaptedThresholds) return;
//    //Now, detect and count the edges;
//    cv::Mat gray,edges;
//    cv::cvtColor(rgb,gray,CV_BGRA2GRAY);
//    const float upFactor = 1.2f, downFactor = 0.8f;
//    int minEdges = 20000;
//    int maxEdges = 25000;
//    cannyThreshold1 = 150;
//    cannyThreshold2= 100;
//    while (!flagAdaptedThresholds)
//    {
//        cv::Canny(gray,edges,static_cast<int>(cannyThreshold1),static_cast<int>(cannyThreshold2),3,true);
//        const int nEdges = cv::countNonZero(edges);
//        I3D_LOG(i3d::info) << "Adapted Canny values: " << cannyThreshold1 << "
//        " << cannyThreshold2 << "nEdges: " << nEdges;
//        //we want to have between 15k and 20k edges
//        if (nEdges >= minEdges && nEdges <= maxEdges)
//            flagAdaptedThresholds = true;
//        else
//        {
//            //if smaller than 15000, lower the threshold else raise it!
//            cannyThreshold1 = (nEdges < minEdges ? cannyThreshold1*downFactor
//            : cannyThreshold1*upFactor); cannyThreshold2 = (nEdges < minEdges
//            ? cannyThreshold2*downFactor : cannyThreshold2*upFactor);
//        }
//    }
//    mPyrConfig.cannyThreshold1 = static_cast<int>(cannyThreshold1);
//    mPyrConfig.cannyThreshold2 = static_cast<int>(cannyThreshold2);
//}
void IOWrapperRGBD::generateImgPyramidFromAstra() {
#ifdef WITH_ORBBEC_ASTRA
  while (this->mSettings.READ_FROM_ASTRA && !mFinish) {
    if (this->orbbecAstraSensor->getImages(rgb, depth,
                                           mSettings.DEPTH_SCALE_FACTOR)) {
      auto start = Timer::getTime();
      nFrames++;
      I3D_LOG(i3d::info) << "nFrames: " << nFrames;
      // there is a strange orbbec bug, where the first two lines of the "color"
      // image are invalid
      rgb.row(2).copyTo(rgb.row(0));
      rgb.row(2).copyTo(rgb.row(1));
      if (mSettings.SKIP_FIRST_N_FRAMES > nFrames) // skips the first n Frames
        continue;
      // if (mSettings.DO_ADAPT_CANNY_VALUES) //try to find the correct Canny
      // values during the first 10 frames
      //    adaptCannyValues();
      if (mSettings.DO_OUTPUT_IMAGES)
        writeImages(rgb, depth, nFrames);
      I3D_LOG(i3d::info) << mSettings.DEPTH_SCALE_FACTOR;
      // mPyrQueue.push(pyr);
      std::unique_ptr<ImgPyramidRGBD> ptrTmp(std::unique_ptr<ImgPyramidRGBD>(
          new ImgPyramidRGBD(mPyrConfig, mCamPyr, rgb, depth, nFrames)));
      {
        std::unique_lock<std::mutex> lock(this->mtx);
        mPyrQueue.push(std::move(ptrTmp));
        I3D_LOG(i3d::error) << "ImgPyramid queued" << mPyrQueue.size();
      }
      auto end = Timer::getTime();
      I3D_LOG(i3d::info) << "Reading image: " << nFrames << " in "
                         << Timer::getTimeDiffMiS(start, end);
    }
    usleep(500);
  }
#else
  I3D_LOG(i3d::error) << "Not compiled with Orbbec Astra support!";
  exit(0);
#endif
}

// #define WITH_REALSENSE
void IOWrapperRGBD::writeImages(const cv::Mat &rgb, const cv::Mat depth,
                                const float timestamp) {
  if (!associateFile.is_open()) {

    auto t = std::time(nullptr);
    auto tm = std::localtime(&t);
    char buffer[80];

    strftime(buffer, 80, "%d-%m-%Y-%I-%M-%S", tm);
    // std::string str(buffer);
    outputImgDir = std::string(buffer);
    if (boost::filesystem::create_directory(outputImgDir)) {
      associateFile.open(outputImgDir + "/associate.txt");
      boost::filesystem::create_directory(outputImgDir + "/rgb");
      boost::filesystem::create_directory(outputImgDir + "/depth");
      // I3D_LOG(i3d::error) << "error generating directories";
    }
  } else {
    I3D_LOG(i3d::error)
        << "Associate file already open! Not generating directories";
  }

  // we create and save to a folder!
  const std::string timestampStr = std::to_string(timestamp);
  const std::string depthFilename =
      outputImgDir + "/depth/" + timestampStr + ".png";
  const std::string rgbFilename =
      outputImgDir + "/rgb/" + timestampStr + ".png";

  // write to associate.txt
  // 1305031471.927651 rgb/1305031471.927651.png 1305031471.924928
  // depth/1305031471.924928.png
  associateFile << timestampStr << " rgb/" + timestampStr + ".png "
                << timestampStr << " depth/" + timestampStr + ".png"
                << std::endl;
  cv::Mat depth16U(depth.rows, depth.cols, CV_16UC1);
  depth.convertTo(depth16U, CV_16U, 1000);
  // save images
  cv::imwrite(depthFilename, depth16U);
  cv::imwrite(rgbFilename, rgb);
}

void IOWrapperRGBD::generateImgPyramidFromRealSense() {

#ifdef WITH_REALSENSE
  while (this->mSettings.READ_FROM_REALSENSE && !mFinish) {
    if (this->realSenseSensor->getImages(rgb, depth,
                                         mSettings.DEPTH_SCALE_FACTOR)) {
      auto start = Timer::getTime();
      nFrames++;
      // there is a strange orbbec bug, where the first two lines of the "color"
      // image are invalid
      if (mSettings.DO_OUTPUT_IMAGES)
        writeImages(rgb, depth, nFrames);
      I3D_LOG(i3d::info) << mSettings.DEPTH_SCALE_FACTOR;
      // mPyrQueue.push(pyr);
      std::unique_ptr<ImgPyramidRGBD> ptrTmp(std::unique_ptr<ImgPyramidRGBD>(
          new ImgPyramidRGBD(mPyrConfig, mCamPyr, rgb, depth, nFrames)));
      {
        std::unique_lock<std::mutex> lock(this->mtx);
        mPyrQueue.push(std::move(ptrTmp));
        I3D_LOG(i3d::error) << "ImgPyramid queued" << mPyrQueue.size();
      }
      auto end = Timer::getTime();
      // if (mPyrQueue.size() > 50) break;
      I3D_LOG(i3d::info) << "Reading image: " << nFrames << " in "
                         << Timer::getTimeDiffMiS(start, end);
    }
    usleep(500);
  }
#else
  I3D_LOG(i3d::error) << "Not compiled with RealSense support!";
  exit(0);
#endif
}

void IOWrapperRGBD::requestQuit() {
  std::unique_lock<std::mutex> lock(this->mtx);
  mQuitFlag = true;
}
void IOWrapperRGBD::generateImgPyramidFromFiles() {
  //    cv::Mat rgb = cv::Mat(480,640,CV_8UC3);
  //    cv::Mat depth = cv::Mat(480,640,CV_32FC1);
  double rgbTimeStamp = 0, depthTimeStamp = 0;

  // Frame count
  std::string line;
  int lineCount = 0;
  fileList.clear();
  fileList.seekg(0, std::ios::beg);
  while (std::getline(fileList, line)) {
    // Ignore lines that start with '#'
    if (line.empty() || line[0] == '#') {
      continue;
    }
    lineCount++;
  }
  fileList.clear();
  fileList.seekg(0, std::ios::beg);
  I3D_LOG(i3d::info) << "Total frames:" << lineCount;

  while (readNextFrame(rgb, depth, rgbTimeStamp, depthTimeStamp,
                       mSettings.SKIP_FIRST_N_FRAMES,
                       mSettings.DEPTH_SCALE_FACTOR) &&
         !mFinish) {
    const double tumRefTimestamp =
        (mSettings.useDepthTimeStamp ? depthTimeStamp : rgbTimeStamp);
    nFrames++;
    I3D_LOG(i3d::info) << "Processing at frame:" << nFrames << "/" << lineCount;
    auto t_start = Timer::getTime();
    I3D_LOG(i3d::trace) << "Before img pyramid!";
    std::unique_ptr<ImgPyramidRGBD> pyrPtr(
        new ImgPyramidRGBD(mPyrConfig, mCamPyr, rgb, depth, tumRefTimestamp));
    I3D_LOG(i3d::trace) << "Creating pyramid: "
                        << Timer::getTimeDiffMiS(t_start, Timer::getTime())
                        << " ms." << mSettings.DEPTH_SCALE_FACTOR;
    {
      auto t_Push = Timer::getTime();
      std::unique_lock<std::mutex> lock(this->mtx);
      mPyrQueue.push(std::move(pyrPtr));
      I3D_LOG(i3d::trace) << "Push wait: "
                          << Timer::getTimeDiffMiS(t_Push, Timer::getTime())
                          << " ms.";
    }
    if (nFrames > mSettings.READ_N_IMAGES)
      break;
    usleep(1000);
  }
  // this->mHasMoreImages = false;
  mAllImagesRead = true;
  while (!mFinish) {
    usleep(3000);
  }
}
bool IOWrapperRGBD::readNextFrame(cv::Mat &rgb, cv::Mat &depth,
                                  double &rgbTimeStamp, double &depthTimeStamp,
                                  int skipFrames, double depthScaleFactor) {
  I3D_LOG(i3d::trace) << "Read next frame.";
  auto start = Timer::getTime();
  bool fileRead = false;
  std::string currRGBFile, currDepthFile;
  std::string inputLine;
  // read lines
  while ((std::getline(fileList, inputLine))) {
    // ignore comments
    if (inputLine[0] == '#' || inputLine.empty())
      continue;
    noFrames++;
    if (noFrames <= skipFrames)
      continue;
    std::istringstream is_associate(inputLine);
    is_associate >> rgbTimeStamp >> currRGBFile >> depthTimeStamp >>
        currDepthFile;
    currRGBFile =
        mSettings.MainFolder + "/" + mSettings.subDataset + "/" + currRGBFile;
    currDepthFile =
        mSettings.MainFolder + "/" + mSettings.subDataset + "/" + currDepthFile;
    I3D_LOG(i3d::debug) << "RGB Files: " << currRGBFile;
    I3D_LOG(i3d::debug) << "Depth Files: " << currDepthFile;
    fileRead = true;
    break;
  }

  if (fileList.eof()) {
    I3D_LOG(i3d::info) << "All frames have been read.";
    return false;
  }

  rgb = cv::imread(currRGBFile);
  depth = cv::imread(currDepthFile, CV_LOAD_IMAGE_UNCHANGED);

  bool exit_flag = false;
  if (rgb.empty()) {
    I3D_LOG(i3d::error) << "Fail to read rgb: " << currRGBFile;
    exit_flag = true;
  }
  if (depth.empty()) {
    I3D_LOG(i3d::error) << "Fail to read depth: " << currDepthFile;
    exit_flag = true;
  }
  if (exit_flag) {
    exit(EXIT_FAILURE);
  }

  depth.convertTo(depth, CV_32FC1, 1.0f / depthScaleFactor);
  // divide by 5000 to get distance in metres
  // depth = depth/depthScaleFactor;
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                Timer::getTime() - start)
                .count();
  I3D_LOG(i3d::trace) << "Read time: " << dt << "ms";
  return fileRead;
}

void IOWrapperRGBD::setFinish(bool setFinish) {
  std::unique_lock<std::mutex> lock(this->mtx);
  this->mFinish = true;
}

bool IOWrapperRGBD::getOldestPyramid(std::shared_ptr<ImgPyramidRGBD> &pyr) {

  I3D_LOG(i3d::trace) << "getOldestPyramid = " << mPyrQueue.size();
  if (mPyrQueue.empty())
    return false;
  I3D_LOG(i3d::trace) << "mPyrQueue.size() = " << mPyrQueue.size();
  std::unique_lock<std::mutex> lock(this->mtx);
  pyr = std::move(mPyrQueue.front());
  mPyrQueue.pop();
  if (mPyrQueue.empty() && mAllImagesRead)
    this->mHasMoreImages = false;
  return pyr != NULL;
}
