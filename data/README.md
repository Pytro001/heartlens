# Data

- `raw/uci411_extension.zip`, `raw/ext/extention of Z-Alizadeh sani dataset.xlsx`: UCI Machine Learning Repository,
  "extention of Z-Alizadeh sani dataset", id 411, DOI 10.24432/C5461K, CC BY 4.0. Downloaded unchanged from
  https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip on 2026-10-07.
  303 rows, 59 columns including the targets LAD, LCX, RCA (Stenotic or Normal) and Cath (CAD or Normal).
  Note: the brief mentioned UCI id 412; on UCI, 412 is the base Z-Alizadeh Sani file (Cath only) and 411 is the
  extension with the per vessel labels. Both zips are kept; only 411 is used.
- `raw/uci412_base.zip`: the base dataset (id 412, DOI 10.24432/C5Q31T), kept for reference.
- `anatomy/stl/*.stl`: BodyParts3D parts (CC BY-SA 2.1 JP) from the GitHub mirror
  https://github.com/Kevin-Mattheus-Moerman/BodyParts3D: FMA7274 wall of heart, FMA3736/3768 aorta, FMA4720 superior
  vena cava, FMA4706/76751 coronary sinus and posterior LV veins, FMA4685 left main, FMA3862nsn + FMA71670 LAD and
  septal branches, FMA3895 circumflex, FMA3802/3818/3840nsn/76994 RCA trunk, marginal, posterior descending and
  posterolateral branches. `make anatomy` decimates them into `web/public/anatomy/heart.glb`.
