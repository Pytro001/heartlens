# HeartLens demo script (film 4:40)

The screen film is `output/heartlens-screen.mp4` (1920x1080, 30 fps, no audio). Put your face cam in a corner and talk over it. Every number below comes from the models running in the page; the cursor and camera are scripted.

## Intro to camera (before the screen starts)

1. "I'm Konstantin. This is HeartLens, my entry for Track A."
2. "It predicts coronary disease and which artery is likely narrowed, from routine clinical data."
3. "And it shows that risk on a 3D heart, with the reasons, in your browser."

## Shot list

| Time | Screen | You say |
|---|---|---|
| 0:00 | Heart slowly turns, arteries grey. Caption "A risk score is a number". | "Cardiologists get a number. Patients get confused. A number doesn't say where." |
| 0:10 | Caption "HeartLens shows where the risk is". | "HeartLens shows where the risk is. This is a real anatomical heart with the three main coronary arteries." |
| 0:20 | Cursor opens the patient picker. Caption "Patient 25, one of 303 real patients". | "These are 303 real patients from a public dataset. Let's pick patient 25. Fifty years old, diabetic, typical chest pain." |
| 0:24 | Arteries light up. Camera swings to the front. | "The arteries light up by predicted stenosis." |
| 0:29 | Caption "Left anterior descending: 91%". LAD glows red. | "Left anterior descending: 91 percent probability of a narrowing." |
| 0:36 | Camera to the left side. "Left circumflex: 23%". | "Left circumflex: low, 23 percent." |
| 0:43 | Camera to the right side. "Right coronary: 39%". | "Right coronary: moderate, 39 percent. Her angiogram showed exactly that pattern: LAD narrowed, the other two normal." |
| 0:50 | Back to front. "Green is low. Red is high." | "Green is low. Red is high." |
| 0:54 | Cursor clicks the LAD label. Camera moves in. LAD card opens on the right. | "Why the LAD? Click it." |
| 1:03 | Caption "SHAP: what pushed this prediction". Waterfall bars. | "Each bar is how much one feature pushed this prediction, in percentage points. The cohort average is 58 percent." |
| 1:14 | Caption "Red raises risk. Green lowers it." | "Typical chest pain adds 17 points. T inversion on the ECG adds about 3. Her age, 50, pulls it down a little." |
| 1:25 | Caption "Compared with all 303 patients". Percentile strip. | "And she is higher than 70 percent of the cohort for the LAD." |
| 1:34 | Cursor grabs the fasting blood sugar slider. | "Now the what-if. What if her blood sugar were controlled?" |
| 1:37 | Slider drags from 180 to 90. Numbers update every frame. | "I drag it from 180 down to 90. The models rerun on every frame." |
| 1:45 | Caption "LAD 91% to 87%. Sugar is not a big driver here." | "The LAD barely moves. In this cohort, blood sugar is not what drives the LAD. The model is honest about that." |
| 1:56 | Cursor toggles typical chest pain off. Caption "Now remove typical chest pain". | "Now remove the typical chest pain." |
| 2:04 | Caption "LAD 91% to 39%. The artery cools." LAD turns green. The card shows "approximate contribution" bars. | "The LAD drops to 39 percent and cools from red to green. These bars are approximate: we put each change back one at a time. The tooltip says exactly that." |
| 2:12 | Cursor clicks "Reset to record". | "Reset. Four models, rerun in under a millisecond, all in the browser." |
| 2:18 | Cursor clicks "Model evidence". Results table. | "How good is it? Five by five repeated cross validation, with tuning inside every fold." |
| 2:30 | Caption "Tuned inside each fold. No target leakage." | "No leakage: when we predict one artery, the other labels are removed. There's a unit test for it." |
| 2:42 | Caption "Three model families compared per target". | "We compare logistic regression, random forest and XGBoost for every target. CAD: ROC AUC 0.93. LAD: 0.85. The circumflex and the right coronary are harder, about 0.74 and 0.72, and we show that." |
| 2:55 | Page scrolls to calibration plots. Caption "Calibrated: predicted matches observed". | "Probabilities are calibrated. When it says 70 percent, about 70 percent of those patients had disease." |
| 3:08 | Cursor clicks "How it works". Architecture diagram. | "Python trains once. Models are exported to ONNX." |
| 3:21 | Caption "ONNX models run on any laptop, no GPU". | "The browser runs them with onnxruntime web. Three.js draws the heart. No server, no GPU." |
| 3:33 | Caption "No server. No data leaves the page." | "Patient data never leaves the page." |
| 3:44 | Cursor clicks "Limitations". | "Limits. 303 patients from one centre in Iran. No external validation yet." |
| 3:59 | Caption "Decision support research only. Not a medical device." | "The colour marks the artery, not the exact spot. And this is decision support research. Not a medical device. The banner never goes away." |
| 4:14 | Back to the patient view. Heart turns slowly. | "Next: a second cohort for external validation, and imaging." |
| 4:17 | Caption "HeartLens. Risk you can see." Hold to the end. | "HeartLens. Risk you can see." |

## Close to camera

1. "Code, report and models are open source under MIT. Links are below."
2. "Thanks for watching."
