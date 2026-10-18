# Limitations

1. **Coarse labels.** CORINE is 100 m with a 25 ha minimum mapping unit; at 10 m, boundaries and small peat patches are
   approximate. Scores measure agreement with CORINE, not with the ground.
2. **Peat is hard.** Peat bogs resemble wet grassland, heath and cutover bog spectrally; confusion with "open semi-natural" is
   expected.
3. **Time mismatch.** Labels are for 2018; imagery year is configurable but still approximate; land cover and peat extraction
   change.
4. **Narrow scope.** One study area, one season, one random block split. No repeated splits or confidence intervals yet.
5. **No field validation** and no peatland-condition or carbon assessment.
6. **Small model.** The U-Net is trained from scratch on a time budget and may not beat the Random Forest.
7. **Proxy labels.** If WorldCover is used, the peat class is a herbaceous-wetland proxy and results are not comparable.
8. **Download step not CI-tested.** CI runs on synthetic arrays; the Planetary Computer download is exercised by running the pipeline.
