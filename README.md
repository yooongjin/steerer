
## test video
```
python tools/test_video.py --cfg configs/QNRF_final.py --pretrained pretrained/QNRF_mae_77.8_mse_138.0.pth --video [video_path]
```

## test image
```
python tools/test_img.py --cfg configs/QNRF_final.py --pretrained pretrained/QNRF_mae_77.8_mse_138.0.pth --img [img_path]
```

## torchscript
```
python tools/torchscript.py --cfg configs/QNRF_final.py --checkpoint pretrained/QNRF_mae_77.8_mse_138.0.pth
```


## Reproduce Counting and Localization Performance

이태원에서 UCF-QNRF 성능이 가장 좋았습니다.


|            | Dataset     |  MAE/MSE |   F1-m./Pre./Rec. (%) | Pretraied Model | Dataset |
|------------|-------- |-------|-------|-------|------|
| This Repo      |  SHHB   | 5.8/8.5 |87.0/89.4/84.82.01 | [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/ET5_eR8n2e5Akm19QvajQJcBTbryGy545hImwr2yzeKMSw?e=J9mwUY)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/Ebo6dbV4hnlCtzFo3S5KW-ABwlCLLYWYADLOyYMGWJ6Qrw?e=L0Y0Wi)|
| This Repo      |  TRANSCOS   | 1.8/3.1 |95.1/91.7/93.4/ | [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EQHeaFzaV_ZAvIdmpbz_lR8BI8a2YzWoka-2Xa__O-O5kA?e=6u8lhT)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EXxeKimCxW1CsP5HjNRlJF8BdfASUGxbBW1q40Ijp_j32A?e=K7cDeZ)|
| This Repo      |  MTC   | 2.74/3.72 |-| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EXolIStQNy9CuxoWo6L6924BpfboWJL1djEfsfENFMohIw?e=7m7fka)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EWIjz_QnX8xAnDKEYS8vgRQBK9MDySll8gmEXxNhxkq2iA?e=jquZdN)|
| This Repo      |  JHU   | 54.5/240.6 |65.6/66.7/64.6| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EYjeF4H3Xw9GlYvtYOhygCEBS7N39Si_izSr9jRH2Pslfg?e=KgIgbe)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/ESXVWJn2zfNHs6x2eOCzJjcB-OdIoRaHeRitYCkmIomyig?e=yrO4IS)|
| This Repo      |  TREE   | 8.2/11.3 |72.9/70.4/75.7| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/ES8QWb_bYZlGgXODD7whQkABueii634dPYvvVtNE9jPlog?e=35331P)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EaciE23qN29LjZPOMkpsm3wB0L_xZaqj-s2Ig2_DMnGFAw?e=fh1IKf)|
| This Repo      |  FDST   | 0.93/1.29 |97.4/98.0/96.7| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/ERU3N-R2bYVPqjWIOpxorcYBTTDPHzkTnj9owFLgQgvURQ?e=SHMpQJ)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EWtkM9DQMKRKhgQBNkxHy64B7AgRsyv8DhnFZRnlrF29Vw?e=Q0VPjG)|
| This Repo      |  UCF-QNRF   | 77.8/138.0 |75.6/79.7/72.0| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/EfE8YRRrAYVBj7HbkC78yPYBPjLURl1ltKlihKhTI1Kl4g?e=yvrPDb)| [Dataset](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/Ef9E9oVtjyBEld_RYpPtqFUBfTBSy6ZgT0rqUhOMgC-X9A?e=WNn9aM)|
| This Repo      |  NWPU   | 32.5/80.4 (Val. set)|-| [weights](https://pjlab-my.sharepoint.cn/:u:/g/personal/hantao_dispatch_pjlab_org_cn/ETu2pnFluOtIozpmfd7ptrUBUvCf2TxUD3_w_aW-9iKX8g?e=DoQNZN)|-||