import logging
import h5py
from pipeline.config import (
    MCAP_FILE_PATH,
    CHUNKS_FILE_PATH,
    MAX_CHUNK_GAP,
    SENSOR_SYNC_THRESHOLD,
    HDF5_WRITE_BATCH_SIZE,
)
from pipeline.reader import MCAPSource
from pipeline.synchronizer import SensorSynchronizer
from pipeline.hdf5_writer import HDF5Writer

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Pipeline")

    source = MCAPSource(MCAP_FILE_PATH)
    synchronizer = SensorSynchronizer(SENSOR_SYNC_THRESHOLD, MAX_CHUNK_GAP)
    writer = HDF5Writer(CHUNKS_FILE_PATH)

    logger.info("Starting conversion pipeline...")
    
    sampleBatch = []
    chunkId = 0
    totalSamples = 0

    try:
        for rawMsg in source.streamMessages():
            for sample in synchronizer.processMessage(rawMsg.topic, rawMsg.msg, rawMsg.timestamp):
                sampleBatch.append(sample)

                if len(sampleBatch) >= HDF5_WRITE_BATCH_SIZE:
                    writer.writeBatch(sampleBatch, chunkId)
                    total_samples += len(sampleBatch)
                    logger.info(f"Processed {total_samples} samples...")
                    sampleBatch = []
                    chunkId += 1

        finalSamples = list(synchronizer.flushSamples())
        if finalSamples:
            sampleBatch.extend(finalSamples)

        if sampleBatch:
            writer.writeBatch(sampleBatch, chunkId)
            total_samples += len(sampleBatch)

        logger.info("Finalizing HDF5 file with metadata...")
        writer.finalize(
            cameraMetadata=source.getCameraMetadata(),
            staticTransforms=synchronizer.staticTransforms
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    
    logger.info("Conversion complete!")

if __name__ == "__main__":
    main()