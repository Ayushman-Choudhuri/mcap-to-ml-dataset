import cv2
import h5py
from mcap_ros2.reader import read_ros2_messages
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

MCAP_FILE_PATH = "data/raw/kitti.mcap"
CHUNKS_FILE_PATH = "data/processed/chunks.hdf5"

MAX_CHUNK_GAP = 0.15
SENSOR_SYNC_THRESHOLD = 0.05
HDF5_WRITE_BATCH_SIZE = 100

TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
LIDAR_TOPIC = "/sensor/lidar/front/points"
CAMERA_INTRINSIC_PARAMETERS_TOPIC = "/sensor/camera/left/camera_info"
CAMERA_IMAGE_TOPIC = "/sensor/camera/left/image_raw/compressed"


SAMPLES_GROUP = "samples"

LIDAR_GROUP = "lidar"
INITIAL_LIDAR_CAPACITY = 1000000
LIDAR_DATA_DATASET_PATH = LIDAR_GROUP + "/data"
LIDAR_OFFSETS_DATASET_PATH = LIDAR_GROUP + "/offsets"
LIDAR_COUNTS_DATASET_PATH = LIDAR_GROUP + "/counts"
LIDAR_POINT_OFFSET_ATTRIBUTE = "lidar_point_offset"
DATA_COMPRESSION_METHOD = "lzf"

CAMERA_GROUP = "camera"
CAMERA_IMAGES_DATASET_PATH = CAMERA_GROUP + "/images"
CAMERA_K_MATRIX_ATTRIBUTE = "camera_k"
CAMERA_D_MATRIX_ATTRIBUTE = "camera_d"
CAMERA_R_MATRIX_ATTRIBUTE = "camera_r"
CAMERA_P_MATRIX_ATTRIBUTE = "camera_p"
DISTORTION_MODEL_ATTRIBUTE = "distortion_model"
CAMERA_WIDTH_ATTRIBUTE = "camera_width"
CAMERA_HEIGHT_ATTRIBUTE = "camera_height"

TRANSFORMS_GROUP = "transforms"
NUM_SAMPLES_ATTRIBUTE = "num_samples"

TIMESTAMP_DATASET_PATH = SAMPLES_GROUP + "/timestamps"
CHUNK_IDS_DATASET_PATH = SAMPLES_GROUP + "/chunk_ids"

TIMESTAMP = "timestamp"
ROS_MSG = "rosMsg"
TF_MATRIX = "tfMatrix"
CHUNK_ID = "chunkId"
LIDAR  = "lidar"
CAMERA = "camera"


class MCAPHandler:
    def __init__(self, mcapFilePath, maxChunkGap=0.15, sensorSyncThreshold=0.05):
        self.mcapFilePath = mcapFilePath
        self.maxChunkGap = maxChunkGap
        self.sensorSyncThreshold = sensorSyncThreshold

        self.staticCache = {
            TF_STATIC_TOPIC: None,
            CAMERA_INTRINSIC_PARAMETERS_TOPIC: None,
        }
        self.chunkBuffer = {
            LIDAR_TOPIC: [],
            CAMERA_IMAGE_TOPIC: [],
            TF_TOPIC: [],
        }
        self.lastTimestamps = {
            LIDAR_TOPIC: None,
            CAMERA_IMAGE_TOPIC: None
        }
        
        self.sampleBuffer = []
        self.tfCache = {}

    def __iter__(self):
        return self.generateSamples()

    def generateSamples(self):
        chunkId = 0
        with open(self.mcapFilePath, "rb") as f:
            for msg in read_ros2_messages(f):
                topic = msg.channel.topic
                rosMsg = msg.ros_msg
                timestamp = msg.log_time.timestamp() 
                
                if hasattr(rosMsg, "header") and hasattr(rosMsg.header, "stamp"):
                    timestamp = rosMsg.header.stamp.sec + (rosMsg.header.stamp.nanosec * 1e-9)
                
                elif topic in [TF_TOPIC, TF_STATIC_TOPIC] and len(rosMsg.transforms) > 0:
                    t0 = rosMsg.transforms[0]
                    timestamp = t0.header.stamp.sec + (t0.header.stamp.nanosec * 1e-9)

                chunkEntry = {
                    TIMESTAMP: timestamp,
                    ROS_MSG: rosMsg,
                }

                if topic in self.staticCache and self.staticCache[topic] is None:
                    self.staticCache[topic] = chunkEntry

                if topic == TF_TOPIC:
                    self.updateTFCache(chunkEntry)
                elif topic == LIDAR_TOPIC:
                    if self.checkFlushConstraint(LIDAR_TOPIC, timestamp):
                        for sample in self.flushSamples():
                            sample[CHUNK_ID] = chunkId
                            yield sample
                        chunkId += 1
                    self.chunkBuffer[LIDAR_TOPIC].append(chunkEntry)
                    self.lastTimestamps[LIDAR_TOPIC] = timestamp
                elif topic == CAMERA_IMAGE_TOPIC:
                    if self.checkFlushConstraint(CAMERA_IMAGE_TOPIC, timestamp):
                        for sample in self.flushSamples():
                            sample[CHUNK_ID] = chunkId
                            yield sample
                        chunkId += 1
                    self.chunkBuffer[CAMERA_IMAGE_TOPIC].append(chunkEntry)
                    self.lastTimestamps[CAMERA_IMAGE_TOPIC] = timestamp
                else:
                    continue
            
            for sample in self.flushSamples():
                sample[CHUNK_ID] = chunkId
                yield sample

    def updateTFCache(self, tfEntry):
        for tfStamped in tfEntry[ROS_MSG].transforms:
            timestamp = tfEntry[TIMESTAMP]
            frameId = tfStamped.header.frame_id
            childFrameId = tfStamped.child_frame_id
            key = f"{frameId}_to_{childFrameId}"
            
            if key not in self.tfCache:
                self.tfCache[key] = []
            
            matrix = self.transformToMatrix(tfStamped.transform)
            self.tfCache[key].append({
                TIMESTAMP: timestamp,
                TF_MATRIX: matrix
            })

    def flushSamples(self):
        lidarFrames = self.chunkBuffer[LIDAR_TOPIC]
        cameraFrames = self.chunkBuffer[CAMERA_IMAGE_TOPIC]
        samples = []

        for lidarEntry in lidarFrames:
            """ Use LIDAR timestamp as the reference """
            lidarTimestamp = lidarEntry[TIMESTAMP]
            closestCamera = self.findClosestFrame(lidarTimestamp, cameraFrames)

            if closestCamera is None:
                continue
            
            timeDiff = abs(lidarTimestamp - closestCamera[TIMESTAMP])

            if timeDiff > self.sensorSyncThreshold:
                continue
            
            transforms = self.interpolateTransforms(lidarTimestamp)
            
            sample = {
                "lidar": lidarEntry,
                "camera": closestCamera,
                "transforms": transforms,
                "timestamp": lidarTimestamp,
            }
            samples.append(sample)
        
        self.chunkBuffer = {
            LIDAR_TOPIC: [],
            CAMERA_IMAGE_TOPIC: [],
            TF_TOPIC: [],
        }
        self.lastTimestamps = {key: None for key in self.lastTimestamps}
        
        for sample in samples:
            yield sample

    def findClosestFrame(self, targetTimestamp, frames):
        if not frames:
            return None
        
        closestFrame = None
        minDiff = float('inf')
        
        for frame in frames:
            diff = abs(frame[TIMESTAMP] - targetTimestamp)
            if diff < minDiff:
                minDiff = diff
                closestFrame = frame
        
        return closestFrame

    def interpolateTransforms(self, targetTimestamp):
        transforms = {}
        
        for key, tfList in self.tfCache.items():
            if not tfList:
                continue
            
            beforeIdx = None
            afterIdx = None
            
            for index, tf in enumerate(tfList):
                if tf[TIMESTAMP] <= targetTimestamp:
                    beforeIdx = index
                if tf[TIMESTAMP] >= targetTimestamp and afterIdx is None:
                    afterIdx = index
                    break
            
            if beforeIdx is not None and afterIdx is not None:
                if beforeIdx == afterIdx:
                    transforms[key] = tfList[beforeIdx][TF_MATRIX]
                else:
                    before = tfList[beforeIdx]
                    after = tfList[afterIdx]
                    
                    alpha = (targetTimestamp - before[TIMESTAMP]) / (after[TIMESTAMP] - before[TIMESTAMP])
                    alpha = np.clip(alpha, 0.0, 1.0)
                    
                    transforms[key] = self.interpolateMatrix(before[TF_MATRIX], after[TF_MATRIX], alpha)
            
            elif beforeIdx is not None:
                transforms[key] = tfList[beforeIdx][TF_MATRIX]
            elif afterIdx is not None:
                transforms[key] = tfList[afterIdx][TF_MATRIX]
        
        return transforms

    def interpolateMatrix(self, matrix1, matrix2, alpha):
        result = np.eye(4, dtype=np.float32)
        
        translation1 = matrix1[0:3, 3]
        translation2 = matrix2[0:3, 3]
        result[0:3, 3] = (1.0 - alpha) * translation1 + alpha * translation2
        
        rotation1 = R.from_matrix(matrix1[0:3, 0:3])
        rotation2 = R.from_matrix(matrix2[0:3, 0:3])
        
        quat1 = rotation1.as_quat()
        quat2 = rotation2.as_quat()
        
        if np.dot(quat1, quat2) < 0:
            quat2 = -quat2
        
        quatInterp = (1.0 - alpha) * quat1 + alpha * quat2
        quatInterp = quatInterp / np.linalg.norm(quatInterp)
        
        result[0:3, 0:3] = R.from_quat(quatInterp).as_matrix().astype(np.float32)
        
        return result

    def checkFlushConstraint(self, sensorTopicName, currentTimestamp):
        lastTimestamp = self.lastTimestamps.get(sensorTopicName)
        if lastTimestamp is None:
            return False
        gap = currentTimestamp - lastTimestamp
        if gap > self.maxChunkGap:
            return True
        return False

    @staticmethod
    def lidarToNumpy(lidarMsg):
        dtypeMap = {
            1: np.int8,
            2: np.uint8,
            3: np.int16,
            4: np.uint16,
            5: np.int32,
            6: np.uint32,
            7: np.float32,
            8: np.float64,
        }

        fields = []
        for field in lidarMsg.fields:
            if field.name in ("x", "y", "z", "intensity"):
                fields.append(
                    (field.name, 
                     dtypeMap[field.datatype])
                )

        if len(fields) < 4:
            raise ValueError("PointCloud2 does not contain x,y,z,intensity")

        cloud = np.frombuffer(lidarMsg.data, dtype=np.dtype(fields))

        return np.stack(
            (cloud["x"], 
             cloud["y"], 
             cloud["z"], 
             cloud["intensity"]
            ),
            axis=-1
        ).astype(np.float32)
    
    @staticmethod
    def compressedImageToNumpy(imageMsg):
        imageArray = np.frombuffer(imageMsg.data, np.uint8)
        cv2Image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        return cv2Image
    
    def transformToMatrix(self, transform):
        translation = transform.translation
        rotation = transform.rotation
        matrix = np.eye(4, dtype=np.float32)
        matrix[0:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]
        matrix[0:3, 0:3] = R.from_quat([
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        ]).as_matrix().astype(np.float32)
        return matrix
    
    def exportSamplesToHDF5(self, samples, h5File):
        numSamples = len(samples)
        
        if numSamples == 0:
            return
        
        startIdx = h5File.attrs.get(NUM_SAMPLES_ATTRIBUTE, 0)
        endIdx = startIdx + numSamples
        
        if "lidar" not in h5File:
            self.createDatasets(h5File, samples[0])
        
        for i, sample in enumerate(samples):
            sampleIdx = startIdx + i
            
            lidarData = self.lidarToNumpy(sample[LIDAR][ROS_MSG])
            cameraData = self.compressedImageToNumpy(sample[CAMERA][ROS_MSG])
            
            h5File[TIMESTAMP_DATASET_PATH][sampleIdx] = sample[TIMESTAMP]
            h5File[CHUNK_IDS_DATASET_PATH][sampleIdx] = sample[CHUNK_ID]
            
            pointOffset = h5File.attrs.get(LIDAR_POINT_OFFSET_ATTRIBUTE, 0)
            numPoints = lidarData.shape[0]
            
            if pointOffset + numPoints > h5File[LIDAR_DATA_DATASET_PATH].shape[0]:
                self.resizeLidarDataset(h5File, pointOffset + numPoints)
            
            h5File[LIDAR_DATA_DATASET_PATH][pointOffset:pointOffset + numPoints] = lidarData
            h5File[LIDAR_OFFSETS_DATASET_PATH][sampleIdx] = pointOffset
            h5File[LIDAR_COUNTS_DATASET_PATH][sampleIdx] = numPoints
            
            h5File.attrs[LIDAR_POINT_OFFSET_ATTRIBUTE] = pointOffset + numPoints
            
            h5File[CAMERA_IMAGES_DATASET_PATH][sampleIdx] = cameraData
            
            for tfKey, tfMatrix in sample["transforms"].items():
                datasetPath = f"transforms/{tfKey}"
                if datasetPath not in h5File:
                    h5File.create_dataset(
                        datasetPath,
                        shape=(0, 4, 4),
                        maxshape=(None, 4, 4),
                        dtype=np.float32,
                        compression="lzf"
                    )
                
                dataset = h5File[datasetPath]
                dataset.resize((sampleIdx + 1, 4, 4))
                dataset[sampleIdx] = tfMatrix
        
        h5File.attrs[NUM_SAMPLES_ATTRIBUTE] = endIdx

    def createDatasets(self, h5File, sampleTemplate):
        h5File.create_group(SAMPLES_GROUP)
        h5File.create_dataset(
            TIMESTAMP_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64
        )
        h5File.create_dataset(
            CHUNK_IDS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32
        )
        
        h5File.create_group(LIDAR_GROUP)
        h5File.create_dataset(
            LIDAR_DATA_DATASET_PATH,
            shape=(INITIAL_LIDAR_CAPACITY, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            compression=DATA_COMPRESSION_METHOD,
            chunks=(10000, 4)
        )
        h5File.create_dataset(
            LIDAR_OFFSETS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64
        )
        h5File.create_dataset(
            LIDAR_COUNTS_DATASET_PATH,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32
        )
        
        h5File.create_group(CAMERA_GROUP)
        cameraData = self.compressedImageToNumpy(sampleTemplate["camera"][ROS_MSG])
        height, width, channels = cameraData.shape
        h5File.create_dataset(
            CAMERA_IMAGES_DATASET_PATH,
            shape=(0, height, width, channels),
            maxshape=(None, height, width, channels),
            dtype=np.uint8,
            compression=DATA_COMPRESSION_METHOD,
            chunks=(1, height, width, channels)
        )
        
        h5File.create_group(TRANSFORMS_GROUP)
        
        h5File.attrs[NUM_SAMPLES_ATTRIBUTE] = 0
        h5File.attrs[LIDAR_POINT_OFFSET_ATTRIBUTE] = 0

    def resizeLidarDataset(self, h5File, newSize):
        currentSize = h5File[LIDAR_DATA_DATASET_PATH].shape[0]
        growthSize = max(newSize - currentSize, currentSize)
        targetSize = currentSize + growthSize
        h5File[LIDAR_DATA_DATASET_PATH].resize((targetSize, 4))

    def resizeDatasets(self, h5File, numNewSamples):
        currentSize = h5File.attrs.get(NUM_SAMPLES_ATTRIBUTE, 0)
        newSize = currentSize + numNewSamples
        
        h5File[TIMESTAMP_DATASET_PATH].resize((newSize,))
        h5File[CHUNK_IDS_DATASET_PATH].resize((newSize,))
        h5File[LIDAR_OFFSETS_DATASET_PATH].resize((newSize,))
        h5File[LIDAR_COUNTS_DATASET_PATH].resize((newSize,))
        h5File[CAMERA_IMAGES_DATASET_PATH].resize((newSize, 
                                                h5File[CAMERA_IMAGES_DATASET_PATH].shape[1], 
                                                h5File[CAMERA_IMAGES_DATASET_PATH].shape[2], 
                                                h5File[CAMERA_IMAGES_DATASET_PATH].shape[3]))
    def finalizeHDF5(self, h5File):
        numSamples = h5File.attrs.get(NUM_SAMPLES_ATTRIBUTE, 0)
        lidarPointOffset = h5File.attrs.get(LIDAR_POINT_OFFSET_ATTRIBUTE, 0)
        
        h5File[LIDAR_DATA_DATASET_PATH].resize((lidarPointOffset, 4))
        
        cameraMetadata = self.staticCache.get(CAMERA_INTRINSIC_PARAMETERS_TOPIC)
        if cameraMetadata:
            msg = cameraMetadata[ROS_MSG]
            h5File.attrs[CAMERA_K_MATRIX_ATTRIBUTE] = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            h5File.attrs[CAMERA_D_MATRIX_ATTRIBUTE] = np.array(msg.d, dtype=np.float32)
            h5File.attrs[CAMERA_R_MATRIX_ATTRIBUTE] = np.array(msg.r, dtype=np.float32).reshape(3, 3)
            h5File.attrs[CAMERA_P_MATRIX_ATTRIBUTE] = np.array(msg.p, dtype=np.float32).reshape(3, 4)
            h5File.attrs[DISTORTION_MODEL_ATTRIBUTE] = str(msg.distortion_model)
            h5File.attrs[CAMERA_WIDTH_ATTRIBUTE] = int(msg.width)
            h5File.attrs[CAMERA_HEIGHT_ATTRIBUTE] = int(msg.height)
        else:
            print("Warning: No Camera Metadata found in the MCAP file!")

        staticTransforms = self.staticCache.get(TF_STATIC_TOPIC)
        if staticTransforms:
            staticGroup = h5File.create_group("static_transforms")
            for tfStamped in staticTransforms[ROS_MSG].transforms:
                matrix = self.transformToMatrix(tfStamped.transform)
                frameId = tfStamped.header.frame_id
                childFrameId = tfStamped.child_frame_id
                key = f"{frameId}_to_{childFrameId}"
                staticGroup.create_dataset(key, data=matrix, dtype=np.float32)
        else:
            print("Warning: No Static TF found in the MCAP file!")
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {numSamples}")
        print(f"  Total lidar points: {lidarPointOffset}")
        print(f"  Average points per sample: {lidarPointOffset / numSamples if numSamples > 0 else 0:.1f}")


if __name__ == "__main__":
    parser = MCAPHandler(MCAP_FILE_PATH, MAX_CHUNK_GAP, SENSOR_SYNC_THRESHOLD)
    
    with h5py.File(CHUNKS_FILE_PATH, "w") as h5File:
        print(f"Starting Conversion...")
        
        sampleBatch = []
        totalSamples = 0
        firstBatch = True
        
        for sample in parser:
            sampleBatch.append(sample)
            
            if len(sampleBatch) >= HDF5_WRITE_BATCH_SIZE:
                if firstBatch:
                    parser.createDatasets(h5File, sampleBatch[0])
                    firstBatch = False
                parser.resizeDatasets(h5File, len(sampleBatch))
                parser.exportSamplesToHDF5(sampleBatch, h5File)
                totalSamples += len(sampleBatch)
                print(f"Processed {totalSamples} samples...")
                sampleBatch = []
        
        """ handling left over samples """
        if sampleBatch:
            if firstBatch:
                parser.createDatasets(h5File, sampleBatch[0])
                firstBatch = False
            parser.resizeDatasets(h5File, len(sampleBatch))
            parser.exportSamplesToHDF5(sampleBatch, h5File)
            totalSamples += len(sampleBatch)
            print(f"Processed {totalSamples} samples...")
        
        print("\nFinalizing HDF5 file...")
        parser.finalizeHDF5(h5File)

    print("\nConversion complete!")