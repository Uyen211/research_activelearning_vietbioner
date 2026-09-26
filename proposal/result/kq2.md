=== Nhánh A (AL) - Vòng lặp 0 (Labeled size: 85/544) ===
  [Thông báo] Vòng < 3: Đã tắt Contextual Masking để ổn định học ranh giới thực thể.
  + Quy mô câu gốc (L_t): 85 câu
  + Quy mô sau Tăng cường (L_aug): 185 câu (+100 câu)
  + Queries trước Negative Sampling: 925
  + Queries sau Negative Sampling: 536 (Tiết kiệm: 42.1%)
    - Queries DƯƠNG TÍNH: 351 | Queries ÂM TÍNH: 185
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 132
    - DiagnosticProcedure: 33
    - Location: 44
    - DateTime: 32
    - Organisation: 21
--------------------------------------------------

config.json: 100%
 781/781 [00:00<00:00, 35.2kB/s]
pytorch_model.bin: 100%
 738M/738M [00:05<00:00, 217MB/s]
LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 39.6175 - Val Loss: 12.2066
Epoch 2 - Train Loss: 28.2440 - Val Loss: 11.3104
Epoch 3 - Train Loss: 23.6324 - Val Loss: 9.8553
Epoch 4 - Train Loss: 21.2316 - Val Loss: 9.3015
Epoch 5 - Train Loss: 17.7333 - Val Loss: 8.3040
Epoch 6 - Train Loss: 14.7134 - Val Loss: 7.8070
Epoch 7 - Train Loss: 12.8167 - Val Loss: 7.4251
Epoch 8 - Train Loss: 11.0250 - Val Loss: 7.0001
Epoch 9 - Train Loss: 11.0731 - Val Loss: 6.8177
Epoch 10 - Train Loss: 9.2495 - Val Loss: 7.5158
Epoch 11 - Train Loss: 8.0807 - Val Loss: 6.9406
Epoch 12 - Train Loss: 8.3710 - Val Loss: 7.5673
Epoch 13 - Train Loss: 7.0342 - Val Loss: 6.6191
Epoch 14 - Train Loss: 6.4863 - Val Loss: 6.5246
Epoch 15 - Train Loss: 7.9917 - Val Loss: 7.4495
Epoch 16 - Train Loss: 7.2181 - Val Loss: 7.1036
Epoch 17 - Train Loss: 5.8315 - Val Loss: 6.7992
Epoch 18 - Train Loss: 5.3795 - Val Loss: 7.0509
Epoch 19 - Train Loss: 4.4756 - Val Loss: 7.3618
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 0 - Test F1-score: 0.4486
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.56      0.65      0.60        23
DiagnosticProcedure       0.22      0.27      0.24        37
           Location       0.52      0.41      0.45        37
       Organisation       0.48      0.50      0.49        22
Symptom_and_Disease       0.51      0.43      0.47       184

          micro avg       0.47      0.43      0.45       303
          macro avg       0.46      0.45      0.45       303
       weighted avg       0.48      0.43      0.45       303

Vòng 0 - Chi phí sửa vòng này: 919 - Tổng chi phí sửa lũy kế: 919

=== Nhánh A (AL) - Vòng lặp 1 (Labeled size: 185/544) ===
  [Thông báo] Vòng < 3: Đã tắt Contextual Masking để ổn định học ranh giới thực thể.
  + Quy mô câu gốc (L_t): 185 câu
  + Quy mô sau Tăng cường (L_aug): 383 câu (+198 câu)
  + Queries trước Negative Sampling: 1915
  + Queries sau Negative Sampling: 1169 (Tiết kiệm: 39.0%)
    - Queries DƯƠNG TÍNH: 793 | Queries ÂM TÍNH: 376
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 384
    - DiagnosticProcedure: 120
    - Location: 90
    - DateTime: 75
    - Organisation: 59
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 45.0196 - Val Loss: 10.2807
Epoch 2 - Train Loss: 29.8422 - Val Loss: 8.0813
Epoch 3 - Train Loss: 19.5504 - Val Loss: 5.9615
Epoch 4 - Train Loss: 15.1754 - Val Loss: 5.1907
Epoch 5 - Train Loss: 12.1240 - Val Loss: 5.0479
Epoch 6 - Train Loss: 11.0625 - Val Loss: 5.3423
Epoch 7 - Train Loss: 9.9675 - Val Loss: 4.9157
Epoch 8 - Train Loss: 8.4649 - Val Loss: 4.8883
Epoch 9 - Train Loss: 8.5981 - Val Loss: 5.4521
Epoch 10 - Train Loss: 7.5610 - Val Loss: 4.8879
Epoch 11 - Train Loss: 6.5631 - Val Loss: 5.2722
Epoch 12 - Train Loss: 5.9067 - Val Loss: 5.2459
Epoch 13 - Train Loss: 6.6668 - Val Loss: 5.1345
Epoch 14 - Train Loss: 5.7031 - Val Loss: 5.4479
Epoch 15 - Train Loss: 5.0960 - Val Loss: 5.1156
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 1 - Test F1-score: 0.5828
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.52      0.70      0.59        23
DiagnosticProcedure       0.32      0.27      0.29        37
           Location       0.57      0.57      0.57        37
       Organisation       0.67      0.45      0.54        22
Symptom_and_Disease       0.64      0.65      0.64       184

          micro avg       0.58      0.58      0.58       303
          macro avg       0.54      0.53      0.53       303
       weighted avg       0.58      0.58      0.58       303

Vòng 1 - Chi phí sửa vòng này: 638 - Tổng chi phí sửa lũy kế: 1557

=== Nhánh A (AL) - Vòng lặp 2 (Labeled size: 285/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 285 câu
  + Quy mô sau Tăng cường (L_aug): 601 câu (+316 câu)
  + Queries trước Negative Sampling: 3005
  + Queries sau Negative Sampling: 1739 (Tiết kiệm: 42.1%)
    - Queries DƯƠNG TÍNH: 1147 | Queries ÂM TÍNH: 592
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 562
    - DiagnosticProcedure: 198
    - Location: 113
    - DateTime: 94
    - Organisation: 77
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 43.5633 - Val Loss: 10.0821
Epoch 2 - Train Loss: 29.8225 - Val Loss: 8.4013
Epoch 3 - Train Loss: 21.8526 - Val Loss: 5.6768
Epoch 4 - Train Loss: 16.6780 - Val Loss: 4.9511
Epoch 5 - Train Loss: 13.6955 - Val Loss: 4.7043
Epoch 6 - Train Loss: 12.1383 - Val Loss: 4.1214
Epoch 7 - Train Loss: 10.8022 - Val Loss: 3.9144
Epoch 8 - Train Loss: 10.2067 - Val Loss: 4.0607
Epoch 9 - Train Loss: 9.2453 - Val Loss: 3.9487
Epoch 10 - Train Loss: 8.3759 - Val Loss: 3.7654
Epoch 11 - Train Loss: 7.7454 - Val Loss: 3.8132
Epoch 12 - Train Loss: 7.2429 - Val Loss: 3.8369
Epoch 13 - Train Loss: 7.0540 - Val Loss: 3.2949
Epoch 14 - Train Loss: 6.7121 - Val Loss: 3.5915
Epoch 15 - Train Loss: 6.0945 - Val Loss: 3.5169
Epoch 16 - Train Loss: 5.9729 - Val Loss: 3.5528
Epoch 17 - Train Loss: 5.8734 - Val Loss: 4.2057
Epoch 18 - Train Loss: 5.6445 - Val Loss: 3.5334
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 2 - Test F1-score: 0.6744
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.66      0.83      0.73        23
DiagnosticProcedure       0.47      0.38      0.42        37
           Location       0.74      0.68      0.70        37
       Organisation       0.61      0.50      0.55        22
Symptom_and_Disease       0.71      0.73      0.72       184

          micro avg       0.68      0.67      0.67       303
          macro avg       0.64      0.62      0.62       303
       weighted avg       0.67      0.67      0.67       303

Vòng 2 - Chi phí sửa vòng này: 423 - Tổng chi phí sửa lũy kế: 1980

=== Nhánh A (AL) - Vòng lặp 3 (Labeled size: 385/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 385 câu
  + Quy mô sau Tăng cường (L_aug): 781 câu (+396 câu)
  + Queries trước Negative Sampling: 3905
  + Queries sau Negative Sampling: 2236 (Tiết kiệm: 42.7%)
    - Queries DƯƠNG TÍNH: 1461 | Queries ÂM TÍNH: 775
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 744
    - DiagnosticProcedure: 247
    - Location: 156
    - DateTime: 125
    - Organisation: 110
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 41.3738 - Val Loss: 10.1316
Epoch 2 - Train Loss: 28.5215 - Val Loss: 7.3338
Epoch 3 - Train Loss: 20.6868 - Val Loss: 5.6727
Epoch 4 - Train Loss: 16.6177 - Val Loss: 5.4304
Epoch 5 - Train Loss: 14.5290 - Val Loss: 5.4528
Epoch 6 - Train Loss: 12.8994 - Val Loss: 4.5960
Epoch 7 - Train Loss: 12.2298 - Val Loss: 4.5743
Epoch 8 - Train Loss: 11.1481 - Val Loss: 4.3390
Epoch 9 - Train Loss: 10.5446 - Val Loss: 4.5157
Epoch 10 - Train Loss: 10.2373 - Val Loss: 4.2080
Epoch 11 - Train Loss: 9.1449 - Val Loss: 3.6858
Epoch 12 - Train Loss: 8.7846 - Val Loss: 3.8750
Epoch 13 - Train Loss: 8.7884 - Val Loss: 3.7293
Epoch 14 - Train Loss: 9.3721 - Val Loss: 4.0573
Epoch 15 - Train Loss: 10.0683 - Val Loss: 3.5216
Epoch 16 - Train Loss: 8.1396 - Val Loss: 3.8194
Epoch 17 - Train Loss: 7.6215 - Val Loss: 3.7576
Epoch 18 - Train Loss: 7.1111 - Val Loss: 3.7566
Epoch 19 - Train Loss: 7.4001 - Val Loss: 3.8503
Epoch 20 - Train Loss: 6.6141 - Val Loss: 3.5109
Epoch 21 - Train Loss: 6.9875 - Val Loss: 3.3469
Epoch 22 - Train Loss: 6.6776 - Val Loss: 3.5393
Epoch 23 - Train Loss: 6.0664 - Val Loss: 3.4784
Epoch 24 - Train Loss: 5.8585 - Val Loss: 3.6900
Epoch 25 - Train Loss: 5.7939 - Val Loss: 3.6808
Vòng 3 - Test F1-score: 0.6316
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.57      0.52      0.55        23
DiagnosticProcedure       0.41      0.49      0.44        37
           Location       0.74      0.62      0.68        37
       Organisation       0.65      0.59      0.62        22
Symptom_and_Disease       0.71      0.65      0.68       184

          micro avg       0.65      0.61      0.63       303
          macro avg       0.62      0.57      0.59       303
       weighted avg       0.66      0.61      0.63       303

Vòng 3 - Chi phí sửa vòng này: 282 - Tổng chi phí sửa lũy kế: 2262

=== Nhánh A (AL) - Vòng lặp 4 (Labeled size: 485/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 485 câu
  + Quy mô sau Tăng cường (L_aug): 952 câu (+467 câu)
  + Queries trước Negative Sampling: 4760
  + Queries sau Negative Sampling: 2621 (Tiết kiệm: 44.9%)
    - Queries DƯƠNG TÍNH: 1679 | Queries ÂM TÍNH: 942
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 876
    - DiagnosticProcedure: 262
    - Location: 178
    - DateTime: 135
    - Organisation: 116
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 36.7585 - Val Loss: 10.1971
Epoch 2 - Train Loss: 25.3500 - Val Loss: 7.8408
Epoch 3 - Train Loss: 19.3706 - Val Loss: 5.8805
Epoch 4 - Train Loss: 15.7277 - Val Loss: 5.0201
Epoch 5 - Train Loss: 13.7858 - Val Loss: 4.5237
Epoch 6 - Train Loss: 12.5398 - Val Loss: 4.6051
Epoch 7 - Train Loss: 11.0638 - Val Loss: 4.4933
Epoch 8 - Train Loss: 10.2804 - Val Loss: 3.9759
Epoch 9 - Train Loss: 9.4039 - Val Loss: 3.7353
Epoch 10 - Train Loss: 8.8670 - Val Loss: 3.5272
Epoch 11 - Train Loss: 8.5405 - Val Loss: 3.4920
Epoch 12 - Train Loss: 7.7801 - Val Loss: 3.3045
Epoch 13 - Train Loss: 7.7863 - Val Loss: 3.4472
Epoch 14 - Train Loss: 7.0993 - Val Loss: 3.5517
Epoch 15 - Train Loss: 6.6956 - Val Loss: 3.3833
Epoch 16 - Train Loss: 6.5171 - Val Loss: 3.2561
Epoch 17 - Train Loss: 6.1537 - Val Loss: 3.5072
Epoch 18 - Train Loss: 5.9640 - Val Loss: 3.4613
Epoch 19 - Train Loss: 5.6773 - Val Loss: 3.1033
Epoch 20 - Train Loss: 5.3778 - Val Loss: 3.3043
Epoch 21 - Train Loss: 5.2636 - Val Loss: 3.3268
Epoch 22 - Train Loss: 5.0573 - Val Loss: 3.5890
Epoch 23 - Train Loss: 4.8304 - Val Loss: 3.3159
Epoch 24 - Train Loss: 4.8121 - Val Loss: 3.4373
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 4 - Test F1-score: 0.7185
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.79      0.83      0.81        23
DiagnosticProcedure       0.53      0.46      0.49        37
           Location       0.65      0.70      0.68        37
       Organisation       0.72      0.59      0.65        22
Symptom_and_Disease       0.76      0.77      0.77       184

          micro avg       0.72      0.72      0.72       303
          macro avg       0.69      0.67      0.68       303
       weighted avg       0.72      0.72      0.72       303

Vòng 4 - Chi phí sửa vòng này: 294 - Tổng chi phí sửa lũy kế: 2556

Nhánh B: Phát hiện checkpoint Vòng 0 từ Nhánh A. Tiến hành nạp baseline đồng bộ...
Nhánh B: Đã đồng bộ F1-score và logs Vòng 0. Bắt đầu chạy lặp từ Vòng 1!

=== Nhánh B (Random) - Vòng lặp 1 (Labeled size: 185/544) ===
  [Thông báo] Vòng < 3: Đã tắt Contextual Masking để ổn định học ranh giới thực thể.
  + Quy mô câu gốc (L_t): 185 câu
  + Quy mô sau Tăng cường (L_aug): 397 câu (+212 câu)
  + Queries trước Negative Sampling: 1985
  + Queries sau Negative Sampling: 1209 (Tiết kiệm: 39.1%)
    - Queries DƯƠNG TÍNH: 822 | Queries ÂM TÍNH: 387
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 384
    - DiagnosticProcedure: 120
    - Location: 90
    - DateTime: 75
    - Organisation: 59
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 46.5978 - Val Loss: 10.2426
Epoch 2 - Train Loss: 32.9895 - Val Loss: 8.1454
Epoch 3 - Train Loss: 23.7262 - Val Loss: 6.4430
Epoch 4 - Train Loss: 18.5487 - Val Loss: 5.9750
Epoch 5 - Train Loss: 15.5510 - Val Loss: 5.4813
Epoch 6 - Train Loss: 12.6611 - Val Loss: 4.9157
Epoch 7 - Train Loss: 10.6917 - Val Loss: 4.8060
Epoch 8 - Train Loss: 9.8394 - Val Loss: 4.5943
Epoch 9 - Train Loss: 8.6768 - Val Loss: 4.5983
Epoch 10 - Train Loss: 7.4233 - Val Loss: 4.9463
Epoch 11 - Train Loss: 6.9034 - Val Loss: 4.6426
Epoch 12 - Train Loss: 6.2601 - Val Loss: 4.9581
Epoch 13 - Train Loss: 5.5969 - Val Loss: 5.0491
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 1 - Test F1-score: 0.5669
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.72      0.78      0.75        23
DiagnosticProcedure       0.26      0.32      0.29        37
           Location       0.71      0.65      0.68        37
       Organisation       0.50      0.64      0.56        22
Symptom_and_Disease       0.57      0.60      0.59       184

          micro avg       0.55      0.59      0.57       303
          macro avg       0.55      0.60      0.57       303
       weighted avg       0.56      0.59      0.57       303

Vòng 1 - Chi phí sửa vòng này: 336 - Tổng chi phí sửa lũy kế: 1255

=== Nhánh B (Random) - Vòng lặp 2 (Labeled size: 285/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 285 câu
  + Quy mô sau Tăng cường (L_aug): 560 câu (+275 câu)
  + Queries trước Negative Sampling: 2800
  + Queries sau Negative Sampling: 1570 (Tiết kiệm: 43.9%)
    - Queries DƯƠNG TÍNH: 1016 | Queries ÂM TÍNH: 554
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 510
    - DiagnosticProcedure: 155
    - Location: 112
    - DateTime: 91
    - Organisation: 65
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 44.1432 - Val Loss: 11.1807
Epoch 2 - Train Loss: 30.6327 - Val Loss: 9.5904
Epoch 3 - Train Loss: 26.0261 - Val Loss: 7.5165
Epoch 4 - Train Loss: 19.8846 - Val Loss: 6.0161
Epoch 5 - Train Loss: 17.1857 - Val Loss: 5.1192
Epoch 6 - Train Loss: 14.5239 - Val Loss: 5.3329
Epoch 7 - Train Loss: 13.4574 - Val Loss: 5.0787
Epoch 8 - Train Loss: 12.2745 - Val Loss: 5.0088
Epoch 9 - Train Loss: 11.5168 - Val Loss: 4.2732
Epoch 10 - Train Loss: 11.9676 - Val Loss: 4.4115
Epoch 11 - Train Loss: 10.0641 - Val Loss: 4.9378
Epoch 12 - Train Loss: 10.3506 - Val Loss: 4.2781
Epoch 13 - Train Loss: 9.3724 - Val Loss: 4.2054
Epoch 14 - Train Loss: 8.6202 - Val Loss: 4.2298
Epoch 15 - Train Loss: 8.2467 - Val Loss: 4.1210
Epoch 16 - Train Loss: 8.0725 - Val Loss: 3.9166
Error displaying widget: model not found
Epoch 17 - Train Loss: 7.8335 - Val Loss: 4.1067
Epoch 18 - Train Loss: 7.2295 - Val Loss: 4.0573
Epoch 19 - Train Loss: 6.9567 - Val Loss: 3.9580
Epoch 20 - Train Loss: 6.2494 - Val Loss: 4.2098
Epoch 21 - Train Loss: 6.3438 - Val Loss: 4.1099
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 2 - Test F1-score: 0.5736
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.59      0.70      0.64        23
DiagnosticProcedure       0.30      0.46      0.37        37
           Location       0.60      0.65      0.62        37
       Organisation       0.35      0.36      0.36        22
Symptom_and_Disease       0.61      0.65      0.63       184

          micro avg       0.54      0.61      0.57       303
          macro avg       0.49      0.56      0.52       303
       weighted avg       0.55      0.61      0.58       303

Vòng 2 - Chi phí sửa vòng này: 242 - Tổng chi phí sửa lũy kế: 1497

=== Nhánh B (Random) - Vòng lặp 3 (Labeled size: 385/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 385 câu
  + Quy mô sau Tăng cường (L_aug): 728 câu (+343 câu)
  + Queries trước Negative Sampling: 3640
  + Queries sau Negative Sampling: 2003 (Tiết kiệm: 45.0%)
    - Queries DƯƠNG TÍNH: 1282 | Queries ÂM TÍNH: 721
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 640
    - DiagnosticProcedure: 186
    - Location: 142
    - DateTime: 106
    - Organisation: 71
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 38.6514 - Val Loss: 10.5667
Epoch 2 - Train Loss: 27.2507 - Val Loss: 9.3445
Epoch 3 - Train Loss: 20.7222 - Val Loss: 6.1689
Epoch 4 - Train Loss: 15.8260 - Val Loss: 5.3818
Epoch 5 - Train Loss: 13.5814 - Val Loss: 5.5089
Epoch 6 - Train Loss: 12.3307 - Val Loss: 5.1422
Epoch 7 - Train Loss: 11.7158 - Val Loss: 4.5954
Epoch 8 - Train Loss: 10.3979 - Val Loss: 4.3708
Epoch 9 - Train Loss: 9.3207 - Val Loss: 5.0980
Epoch 10 - Train Loss: 9.1790 - Val Loss: 4.3813
Epoch 11 - Train Loss: 8.4999 - Val Loss: 4.1300
Epoch 12 - Train Loss: 7.6821 - Val Loss: 4.0269
Epoch 13 - Train Loss: 7.6166 - Val Loss: 4.0641
Epoch 14 - Train Loss: 6.9460 - Val Loss: 4.0186
Epoch 15 - Train Loss: 6.7454 - Val Loss: 4.4481
Epoch 16 - Train Loss: 6.5040 - Val Loss: 3.7472
Epoch 17 - Train Loss: 6.2018 - Val Loss: 3.6080
Epoch 18 - Train Loss: 5.9923 - Val Loss: 4.1073
Epoch 19 - Train Loss: 5.7555 - Val Loss: 3.8704
Epoch 20 - Train Loss: 5.3781 - Val Loss: 3.6044
Epoch 21 - Train Loss: 5.4180 - Val Loss: 3.7739
Epoch 22 - Train Loss: 5.0063 - Val Loss: 3.6574
Epoch 23 - Train Loss: 5.1274 - Val Loss: 3.6945
Epoch 24 - Train Loss: 4.8402 - Val Loss: 4.0189
Epoch 25 - Train Loss: 4.7044 - Val Loss: 3.4871
Vòng 3 - Test F1-score: 0.6816
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.53      0.74      0.62        23
DiagnosticProcedure       0.41      0.41      0.41        37
           Location       0.64      0.68      0.66        37
       Organisation       0.61      0.64      0.62        22
Symptom_and_Disease       0.74      0.77      0.76       184

          micro avg       0.66      0.70      0.68       303
          macro avg       0.59      0.65      0.61       303
       weighted avg       0.66      0.70      0.68       303

Vòng 3 - Chi phí sửa vòng này: 224 - Tổng chi phí sửa lũy kế: 1721

=== Nhánh B (Random) - Vòng lặp 4 (Labeled size: 485/544) ===
  [Thông báo] Vòng >= 2: Đã kích hoạt lại Contextual Masking (Entity: 18%, Context: 15%).
  + Quy mô câu gốc (L_t): 485 câu
  + Quy mô sau Tăng cường (L_aug): 937 câu (+452 câu)
  + Queries trước Negative Sampling: 4685
  + Queries sau Negative Sampling: 2507 (Tiết kiệm: 46.5%)
    - Queries DƯƠNG TÍNH: 1578 | Queries ÂM TÍNH: 929
  + Số lượng thực thể gốc được gán nhãn trong pool:
    - Symptom_and_Disease: 763
    - DiagnosticProcedure: 208
    - Location: 153
    - DateTime: 115
    - Organisation: 83
--------------------------------------------------

LoRA Adapter successfully integrated into DeBERTa backbone.
trainable params: 589,824 || all params: 184,344,576 || trainable%: 0.3200
Epoch 1 - Train Loss: 35.3081 - Val Loss: 10.3354
Epoch 2 - Train Loss: 24.6456 - Val Loss: 8.4930
Epoch 3 - Train Loss: 20.7683 - Val Loss: 7.3434
Epoch 4 - Train Loss: 15.8959 - Val Loss: 5.6525
Epoch 5 - Train Loss: 13.4523 - Val Loss: 5.2921
Epoch 6 - Train Loss: 11.6483 - Val Loss: 4.7114
Epoch 7 - Train Loss: 10.4288 - Val Loss: 5.1419
Epoch 8 - Train Loss: 9.7593 - Val Loss: 4.2414
Epoch 9 - Train Loss: 8.8034 - Val Loss: 3.9734
Epoch 10 - Train Loss: 7.9409 - Val Loss: 4.0254
Epoch 11 - Train Loss: 7.6971 - Val Loss: 3.6180
Epoch 12 - Train Loss: 7.2381 - Val Loss: 3.8038
Epoch 13 - Train Loss: 6.8279 - Val Loss: 3.4451
Epoch 14 - Train Loss: 6.5845 - Val Loss: 3.4729
Epoch 15 - Train Loss: 6.1383 - Val Loss: 3.6317
Epoch 16 - Train Loss: 5.9772 - Val Loss: 3.8482
Epoch 17 - Train Loss: 5.5721 - Val Loss: 3.5927
Epoch 18 - Train Loss: 5.3122 - Val Loss: 3.4834
Dừng sớm (Early Stopping) được kích hoạt!
Vòng 4 - Test F1-score: 0.6307
Class-wise Evaluation Report:
                     precision    recall  f1-score   support

           DateTime       0.73      0.83      0.78        23
DiagnosticProcedure       0.37      0.46      0.41        37
           Location       0.53      0.62      0.57        37
       Organisation       0.50      0.64      0.56        22
Symptom_and_Disease       0.68      0.68      0.68       184

          micro avg       0.61      0.66      0.63       303
          macro avg       0.56      0.65      0.60       303
       weighted avg       0.62      0.66      0.63       303

Vòng 4 - Chi phí sửa vòng này: 246 - Tổng chi phí sửa lũy kế: 1967
WARNING: Ngân sách cạn kiệt
Đã chạy xong/tiếp tục Nhánh B.
