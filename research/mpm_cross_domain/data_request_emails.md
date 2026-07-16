# MPM 类器官图像数据请求邮件模板

> **日期**：2026-07-15
> **作者**：曼卿（Organoid-FL Agent）
> **目的**：冬生审核后发出 4 封邮件给 MPM PDO 论文作者，获取原始图像数据
> **注意**：英文邮件，礼貌简洁，附上合作邀请

---

## 邮件 1：Sarah Best（bioRxiv 2026，44 PDO，最大队列）

**To**: sarah.best@sydney.edu.au
**Subject**: Request for MPM organoid image data — AI cross-domain detection study (XJTLU)

Dear Dr. Best,

I am Dechang Xu, a professor at Xi'an Jiaotong-Liverpool University (XJTLU), leading a research program on AI-driven organoid image analysis for malignant pleural mesothelioma (MPM). Our team is collaborating with Shanghai Jiao Tong University Ruijin Hospital on a medical-AI cross-disciplinary project to build an MPM organoid multimodal data quality assessment prototype.

Your recent bioRxiv preprint *"Patient-derived organoids from malignant pleural effusion"* (biorxiv.org/content/10.64898/2026.03.04.709498) — establishing 44 PDOs from 33 patients — is the largest MPM PDO cohort reported to date. We were particularly impressed by the bright-field images of MPM-PDOs in Figure 1.

We are writing to respectfully request access to:
1. **Bright-field microscopy images** of the 44 MPM-PDOs (or a subset if data sharing is constrained)
2. **Associated clinical annotations** if available: patient demographics, MPM subtype (epithelioid/sarcomatoid/biphasic), culture time, drug response data
3. **Resolution/format**: original TIFF or PNG preferred; de-identified per your institution's protocols

**Purpose and use**:
- Build the first public MPM organoid AI benchmark (current public datasets: 0)
- Train and validate cross-domain detection models (we have a mouse-liver-trained RF-DETR achieving 89% mAP that we plan to fine-tune on MPM)
- Preliminary zero-shot experiments show our model fails on MPM (domain gap 0.272 in ResNet feature space), indicating real MPM data is essential

**What we offer in return**:
- We will sign a Material Transfer Agreement (MTA) following your institution's policy
- All resulting AI models trained on your data will be shared back with your lab
- Co-authorship on the first paper using your data (subject to your contribution)
- Acknowledgment in all publications and grant proposals

**Our team credentials**:
- I have h-index 9, 26 publications, prior experience in federated learning and privacy-preserving medical AI (https://github.com/dechang64)
- Our lab has GPU compute (RTX 3060 + cloud A100 access) and Streamlit-based AI platform (organoid-fl) ready for analysis
- The Ruijin Hospital collaboration ensures clinical translation

We are happy to schedule a video call to discuss data formats, sharing protocols, and potential collaboration. Please let me know if this is feasible.

Looking forward to your response.

Best regards,

**Dechang Xu, PhD**
Professor, Intelligent Science Department
School of Advanced Technology
Xi'an Jiaotong-Liverpool University
Suzhou, Jiangsu, China
Email: dechang.xu@xjtlu.edu.cn

---

## 邮件 2：Li Chenggang（Sci Rep 2025，11 PDO，已有合作基础）

**To**: (从论文查找通讯邮箱)
**Subject**: Request for additional MM PDO image data — cross-domain AI validation

Dear Prof. Li,

I am Dechang Xu from Xi'an Jiaotong-Liverpool University (XJTLU). I have been following your team's work on malignant mesothelioma organoids — your recent *Scientific Reports* paper (Liu Y et al., 2025, DOI: 10.1038/s41598-025-25087-0) on the MM PDO-T cell co-culture platform is a remarkable contribution.

Our team at XJTLU is currently developing AI-driven organoid image analysis tools, and we have already begun extracting bright-field images from Figure 1 of your published paper for our preliminary analysis. However, the published images are limited to figure panels; for training robust AI models, we would benefit from access to the original raw microscopy images.

**We would like to request**:
1. **Original bright-field images** of the 11 MM PDOs (higher resolution than published)
2. **H&E and IHC images** if available in raw format (CK5/6, D2-40, WT-1, CR, CK7, Ki67)
3. **Time-course images** (day 0, 3, 7, 14 of culture, if collected)
4. **NGS data** is already public via GSA-Human HRA011843, no need to re-share

**Use case**:
- Train cross-domain detection/segmentation models for MPM organoids
- Develop quality assessment tools for MPM PDO culture monitoring
- Build a publicly available MPM organoid AI benchmark (currently 0 public datasets)

**In return**:
- Co-authorship on AI-related publications using your data
- Share all trained models with your lab
- Acknowledge your contribution in all grant proposals (currently applying for NSFC and Ruijin Hospital cross-disciplinary grant)

Given our shared interest in advancing MPM research, I would be delighted to discuss this further. Our team is also interested in extending your PDO-T cell co-culture platform with AI-based viability scoring.

Looking forward to hearing from you.

Sincerely,

**Dechang Xu, PhD**
Professor, Intelligent Science Department
Xi'an Jiaotong-Liverpool University
Email: dechang.xu@xjtlu.edu.cn

---

## 邮件 3：Marco Falasca（Lung Cancer 2024，Flinders）

**To**: marco.falasca@nottingham.ac.uk
**Subject**: Request for mesothelioma PDO image data — cross-domain AI validation

Dear Prof. Falasca,

I am Dechang Xu, a professor at Xi'an Jiaotong-Liverpool University (XJTLU) in Suzhou, China. I lead a research program on AI-driven organoid image analysis, with current focus on malignant pleural mesothelioma (MPM).

Your recent publication in *Lung Cancer* (Hocking et al., 2024, "Establishing mesothelioma patient-derived organoid models from malignant pleural effusions") is a pioneering work establishing long-term MPM organoid cultures from pleural effusions.

Our team is collaborating with Ruijin Hospital (Shanghai) on building an MPM organoid multimodal AI prototype, and we would benefit from access to your PDO image data for cross-domain validation.

**Specifically, we request**:
1. **Bright-field images** of the mesothelioma organoid cultures (long-term cultures)
2. **Histology images** (H&E if available)
3. **IHC images** for mesothelioma markers (AE1/AE3, D2-40, CK5/6, WT1, vimentin)
4. **Time-course images** showing organoid growth and morphology

**Why this matters**:
- Public MPM organoid datasets are currently zero
- Our preliminary cross-domain experiments (mouse-liver-trained RF-DETR → MPM images) show domain gap of 0.272 in feature space, indicating real MPM data is essential
- Your data would be the first systematic MPM organoid AI benchmark

**What we offer**:
- Material Transfer Agreement per your institution
- Co-authorship on resulting publications
- Trained AI models shared back
- Acknowledgment in NSFC grant application

Would you be available for a brief video call to discuss data sharing? We are flexible on timing to accommodate your time zone (UK/Australia).

Best regards,

**Dechang Xu, PhD**
Professor, Intelligent Science Department
Xi'an Jiaotong-Liverpool University
Suzhou, China
Email: dechang.xu@xjtlu.edu.cn

---

## 邮件 4：Sarah Knox（BMC Cancer 2023，3 MM organoid lines）

**To**: sarah.knox@ucsf.edu
**Subject**: Inquiry regarding MM organoid image data from BMC Cancer 2023

Dear Dr. Knox,

I am Dechang Xu from Xi'an Jiaotong-Liverpool University (XJTLU) in Suzhou, China. I lead an AI research program for organoid image analysis, currently focused on malignant mesothelioma.

Your *BMC Cancer* publication (Ito et al., 2023, "Matrigel-based organoid culture of malignant mesothelioma reproduces cisplatin sensitivity through CTR1") provided clear bright-field images of MM organoids at 7 days (Figure 2a) that we found particularly useful for our preliminary AI analysis.

Our team is building a cross-domain organoid detection benchmark, and the 3 MM organoid lines you established would be valuable additions. We have already extracted the published Figure 2a panels, but access to the original raw images would significantly improve our analysis.

**Specifically, we would appreciate**:
1. **Original bright-field images** of the 3 MM organoid lines (m5, m107, etc.)
2. **Time-course images** if available (7-day, 14-day, etc.)
3. **Confocal microscopy images** if available (the paper mentions Zeiss LSM 880 with Airyscan)

**Use case**:
- MPM organoid AI benchmark construction
- Cross-domain detection model fine-tuning
- We have demonstrated that SAM2 (segment-anything-2) achieves IoU 0.952 on MPM images with RF-DETR box prompts, suggesting SAM2 mask as pseudo-GT is a viable strategy

**In return**:
- Co-authorship on resulting publications
- Trained models shared back
- Full acknowledgment in grant applications

Thank you for your time. Please let me know if data sharing is feasible, or if there are specific protocols we should follow.

Sincerely,

**Dechang Xu, PhD**
Professor, Intelligent Science Department
Xi'an Jiaotong-Liverpool University
Suzhou, China
Email: dechang.xu@xjtlu.edu.cn

---

## 邮件发送策略

| # | 收件人 | 优先级 | 发送时机 | 预期回复时间 | 预期产出 |
|---|--------|--------|---------|-------------|---------|
| 1 | Sarah Best | **P0** | 本周 | 2-4 周 | 44 PDO（最大队列）|
| 2 | Li Chenggang | P0 | 本周 | 1-2 周 | 11 PDO（已有合作基础）|
| 3 | Marco Falasca | P1 | 本周 | 2-4 周 | 长期培养 MPM PDO |
| 4 | Sarah Knox | P2 | 下周 | 2-4 周 | 3 MM organoid lines |

**回复率预估**：30-50%（学术数据共享请求的典型回复率）

**Fallback 计划**：
- 如果 2 周无回复：发送 follow-up 邮件
- 如果 4 周无回复：尝试联系论文共同作者
- 如果拒绝：请求只共享 de-identified 子集
- 全部失败：依赖瑞金医工交叉项目（2027.Q3）

---

## 邮件附件建议

每封邮件附上：
1. **PI 简介 PDF**（一页 CV）
2. **团队代表性论文**（1-2 篇）
3. **预实验结果摘要**（一页）：
   - RF-DETR zero-shot: median conf 0.014
   - ResNet domain gap: 0.272
   - SAM2 IoU: 0.952
   - 这些数字能证明你的研究是认真的，不是随便要数据

---

## 2026-07-16 更新：加入三层证据链结果

**今天完成的三层证据链**（已 commit + push GitHub，可在附件中引用）：

| 层级 | 数据 | 样本 | Zero-shot AUC | Distilled AUC | 提升 |
|---|---|---|---|---|---|
| Simulation | Intestinal 500+200 | 700 | 0.6616 | 0.9437 | **+28.22%** |
| **真实跨域** | **鼠肝 RF-DETR → Intestinal 50+30** | **10578** | **0.7480** | **0.8857** | **+13.77%** |
| MPM 试点 | MPM 34 patches | 3131 | 0.9779 | 1.0000 | +2.21% |

**关键信息可加入邮件**：
> "We have validated the self-distillation method on three tiers:
> (1) Simulation with 500+200 intestinal organoid samples: AUC improved +28.22%
> (2) Real cross-domain (mouse liver → intestinal, 10,578 samples): +13.77%
> (3) MPM pilot (34 patches, 10 positives): pipeline runnable, sample-limited
>
> All code is open-source at https://github.com/dechang64/organoid-fl
> (commits 788f442, 258a0f9, ada4ac8 on 2026-07-16)."

**GitHub commit references** (可作为邮件附件链接)：
- 788f442 — SAM2 自蒸馏原始脚本（MPM 34 patches）
- 258a0f9 — Intestinal simulation +28.22%
- ada4ac8 — 真实跨域（鼠肝→Intestinal）+13.77%

---

## 邮件发送前 checklist

- [ ] 冬生确认 4 个邮箱地址（Sarah Best / Li Chenggang / Marco Falasca / Sarah Knox）
- [ ] 冬生确认 XJTLU 邮箱可用（dechang.xu@xjtlu.edu.cn）
- [ ] 附件 1：PI 一页 CV PDF
- [ ] 附件 2：团队代表性论文 1-2 篇 PDF
- [ ] 附件 3：预实验结果摘要（含三层证据链）
- [ ] 邮件 2 (Li Chenggang) 需要从 Sci Rep 2025 论文查通讯邮箱
- [ ] 发送后记录日期，建立 follow-up 日历提醒

---

**文档版本**：v1.1（2026-07-16 更新三层证据链）
**状态**：待冬生审核邮箱地址和内容后发出
