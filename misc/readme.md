
## Finding a needle in Haystack Facebook’s photo storage
- "We carefully reduce this per photo metadata so that Haystack storage machines can perform all metadata lookups in main memory. This choice conserves disk operations for reading actual data and thus increases overall throughput."
    - "accessing metadata is the throughput bottleneck"
- "Haystack is an object store that we designed for sharing photos on Facebook where data is written once, read often, never modified, and rarely deleted."
- "large number of requests for less popular (often older) content, which we refer to as the long tail"
- "After reducing directory sizes to hundreds of images per directory, the resulting system would still generally incur 3 disk operations to fetch an image: one to read the directory metadata into memory, a second to load the inode into memory, and a third to read the file contents."
- "The system just needs enough main memory so that all of the filesyste metadata can be cached at once." "we decided to build a custom storage system that reduces the amount of filesystem metadata per photo so that having enough main memory is dramatically more cost-effective than buying more NAS appliances"
- "storing a single photo per file resulted in more filesystem metadata than could be reasonably cached" "Haystack takes a straight-forward approach: it stores multiple photos in a single file and therefore maintains very large files."
- "We organize the Cache as a distributed hash table and use a photo’s id as the key to locate cached data."
- "A Store machine can access a photo quickly using only the id of the corresponding logical volume and the file offset at which the photo resides. This knowledge is the keystone of the Haystack design: retrieving the filename, offset, and size for a particular photo without needing disk operations."
- "A Store machine represents a physical volume as a large file consisting of a superblock followed by a sequence of needles. Each needle represents a photo stored in Haystack."

**Takeaways**
- read, write, and delete.
- keep the index in memory. update asynchronously. use compaction to free up space.

