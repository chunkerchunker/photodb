# Future work

- consider using mivolov2_d1_384 (instead of mivolo_d1)-- slower but should give much more accurate age estimates
- since we've decoupled MiVOLO/YOLO, try YOLO26 now?

- Migrate to YOLO26x once well tested with MiVOLO (much faster; should work as is with MiVOLO, but not well tested)
- Migrate to [SCGNet](https://github.com/happpyjsy/SCGNet) once there are published implementations
- Or at least migrate to MPS/CoreML version of MiVOLO once available (current version runs slower on MPS than CPU)
