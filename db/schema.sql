-- =============================================================================
--  MOOC 辍学风险预测系统 · 数据库脚本（表结构 + 数据字典 + 样本数据）
--  文档编号：MOOC-DROPOUT_DB_001        版本：V1.0        日期：2026-11-04
--  作者：廖祥鳌（2350341066）    指导老师：王明鑫
-- -----------------------------------------------------------------------------
--  【为什么有这个脚本】
--    本系统在实现上不使用关系型数据库：模型、阈值、特征顺序以文件落盘
--    （artifacts/model/ 下的 model.joblib、metadata.json 等），预测结果不持久化。
--    按课程交付规范要求提供 .sql，故本脚本给出与实现**完全等价**的关系模型：
--      ① 表结构：把文件契约（模型产物 / 特征向量 / 预测结果）翻译成 8 张表；
--      ② 数据字典：feature_dict 收录全部 44 维特征的分组与业务含义；
--      ③ 样本数据：课程、选课、行为事件、模型版本、一条真实预测记录。
--    这些数据全部可从交付包内文件复核：
--      模型版本/阈值/指标  → artifacts/realdata_official/model/metadata.json
--      特征字典            → GET /v1/model/features
--      行为事件            → data/sample/request.json
--      预测记录            → 对该请求调用 POST /v1/predict 的真实响应
--    迁移到 MySQL 8.0+ 即可支持多课程、多学期、多模型版本的长期管理。
--    数据库：MySQL 8.0（utf8mb4）；PostgreSQL 需把 AUTO_INCREMENT 改为 GENERATED
--    和 TINYINT 改为 SMALLINT，其余语句可直接使用。
-- =============================================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

CREATE DATABASE IF NOT EXISTS mooc_dropout
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE mooc_dropout;

DROP TABLE IF EXISTS intervention_list;
DROP TABLE IF EXISTS prediction_log;
DROP TABLE IF EXISTS feature_vector;
DROP TABLE IF EXISTS feature_dict;
DROP TABLE IF EXISTS behavior_event;
DROP TABLE IF EXISTS enrollment;
DROP TABLE IF EXISTS model_artifact;
DROP TABLE IF EXISTS course;

-- -----------------------------------------------------------------------------
-- 1. 课程表：一门 MOOC 课程，观察窗为其开课后的前 N 天
-- -----------------------------------------------------------------------------
CREATE TABLE course (
  course_id     VARCHAR(32)  NOT NULL COMMENT '课程编号，与平台一致',
  course_name   VARCHAR(128) NOT NULL COMMENT '课程名称',
  start_date    DATE         NOT NULL COMMENT '开课日期',
  window_days   SMALLINT     NOT NULL DEFAULT 14 COMMENT '观察窗长度（天），本项目固定 14',
  PRIMARY KEY (course_id)
) ENGINE=InnoDB COMMENT='课程';

-- -----------------------------------------------------------------------------
-- 2. 选课表：一行 = 一名学员的一门课；label_dropout 为官方辍学标签（0/1）
-- -----------------------------------------------------------------------------
CREATE TABLE enrollment (
  enrollment_id   VARCHAR(32) NOT NULL COMMENT '选课记录编号',
  course_id       VARCHAR(32) NOT NULL COMMENT '所属课程',
  username        VARCHAR(64) NOT NULL COMMENT '学习者 ID（脱敏后的稳定标识）',
  label_dropout   TINYINT     NULL     COMMENT '是否辍学：1=是，0=否，NULL=未标注',
  split_name      ENUM('train','test') NULL COMMENT '训练/测试切分，按 username 分组切分',
  PRIMARY KEY (enrollment_id),
  KEY idx_enroll_course (course_id),
  KEY idx_enroll_user (username),
  CONSTRAINT fk_enroll_course FOREIGN KEY (course_id) REFERENCES course (course_id)
) ENGINE=InnoDB COMMENT='选课记录';

-- -----------------------------------------------------------------------------
-- 3. 行为事件表：KDD Cup 2015 的原始日志格式（time / source / event / object）
--    真实数据约 1900 万行；线上库按 enrollment_id + event_time 建复合索引
-- -----------------------------------------------------------------------------
CREATE TABLE behavior_event (
  event_id      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  enrollment_id VARCHAR(32) NOT NULL COMMENT '所属选课记录',
  event_time    DATETIME    NOT NULL COMMENT '事件发生时间',
  source        ENUM('access','discussion','navigate','page_close','problem','video','wiki')
                            NOT NULL COMMENT '事件来源，与服务端校验枚举一致',
  event         VARCHAR(32) NOT NULL COMMENT '动作，如 play_video / submit',
  object        VARCHAR(64) NULL     COMMENT '动作对象（视频/习题/页面编号）',
  PRIMARY KEY (event_id),
  KEY idx_event_enroll_time (enrollment_id, event_time),
  CONSTRAINT fk_event_enroll FOREIGN KEY (enrollment_id) REFERENCES enrollment (enrollment_id)
) ENGINE=InnoDB COMMENT='行为事件原始日志';

-- -----------------------------------------------------------------------------
-- 4. 特征字典表：44 维特征的分组与业务含义，即《需求规格说明书》3.5 节的数据字典
--    内容来自 GET /v1/model/features，与模型训练时的列顺序 f_ordinal 一一对应
-- -----------------------------------------------------------------------------
CREATE TABLE feature_dict (
  f_ordinal    SMALLINT     NOT NULL COMMENT '列序号 1..44，与 model.joblib 的 feature_names 顺序一致',
  feature_name VARCHAR(48)  NOT NULL COMMENT '特征名（英文，代码中用）',
  f_group      VARCHAR(24)  NOT NULL COMMENT '特征分组',
  description  VARCHAR(160) NOT NULL COMMENT '业务含义（中文）',
  PRIMARY KEY (f_ordinal),
  UNIQUE KEY uk_feature_name (feature_name)
) ENGINE=InnoDB COMMENT='44 维特征字典';

INSERT INTO feature_dict (f_ordinal, feature_name, f_group, description) VALUES
  (1, 'total_events', '基础活跃', '观察窗内行为事件总数'),
  (2, 'active_days', '基础活跃', '有行为记录的 distinct 天数'),
  (3, 'unique_objects', '基础活跃', '触及的 distinct 学习对象数'),
  (4, 'span_days', '基础活跃', '首个行为到最后一个行为的天数跨度'),
  (5, 'days_since_last_event', '基础活跃', '距观察窗结束的天数（越大说明越早沉默）'),
  (6, 'events_per_active_day', '基础活跃', '活跃日均事件数'),
  (7, 'events_per_day', '基础活跃', '按窗口长度折算的日均事件数'),
  (8, 'active_day_ratio', '基础活跃', '活跃天数占观察窗比例'),
  (9, 'span_ratio', '基础活跃', '行为跨度占观察窗比例'),
  (10, 'n_video', '视频行为', '视频类事件数'),
  (11, 'n_problem', '习题行为', '习题类事件数'),
  (12, 'n_discussion', '论坛行为', '论坛类事件数'),
  (13, 'n_navigate', '导航行为', '页面导航事件数'),
  (14, 'n_page_close', '导航行为', '页面关闭事件数'),
  (15, 'n_access', '导航行为', '课程访问事件数'),
  (16, 'n_wiki', '导航行为', 'wiki 浏览事件数'),
  (17, 'video_ratio', '视频行为', '视频事件占全部事件比例'),
  (18, 'problem_ratio', '习题行为', '习题事件占比'),
  (19, 'discussion_ratio', '论坛行为', '论坛事件占比'),
  (20, 'navigate_ratio', '导航行为', '导航事件占比'),
  (21, 'video_unique_objects', '视频行为', '观看过的 distinct 视频数'),
  (22, 'video_replay_ratio', '视频行为', '视频播放次数 / 去重视频数（重复观看强度）'),
  (23, 'problem_unique_objects', '习题行为', '作答过的 distinct 题目数'),
  (24, 'problem_repeat_ratio', '习题行为', '习题提交次数 / 去重题目数（反复尝试强度）'),
  (25, 'discussion_unique_objects', '论坛行为', '参与的 distinct 讨论区/帖子数'),
  (26, 'night_event_ratio', '时间节律', '22:00-06:00 事件占比'),
  (27, 'weekend_event_ratio', '时间节律', '周末事件占比'),
  (28, 'hour_entropy', '时间节律', '行为小时分布的香农熵，衡量学习节律规律性'),
  (29, 'n_sessions', '会话结构', '按 30 分钟间隔切分的学习会话数'),
  (30, 'mean_session_events', '会话结构', '平均每个会话的事件数'),
  (31, 'max_session_events', '会话结构', '最长一次会话的事件数'),
  (32, 'mean_gap_hours', '会话结构', '相邻事件平均间隔（小时）'),
  (33, 'gap_std_hours', '会话结构', '相邻事件间隔标准差（小时）'),
  (34, 'max_gap_days', '会话结构', '最长一次中断的天数'),
  (35, 'decayed_events', '趋势衰减', '按 exp(-λ·剩余天数) 加权后的事件量'),
  (36, 'decayed_active_days', '趋势衰减', '按时间衰减加权后的活跃天数'),
  (37, 'early_events', '趋势衰减', '窗口前半段事件数'),
  (38, 'late_events', '趋势衰减', '窗口后半段事件数'),
  (39, 'late_event_ratio', '趋势衰减', '后半段事件占比，衡量学习持续性'),
  (40, 'activity_trend', '趋势衰减', '(后半 - 前半) / (后半 + 前半)，取值 [-1, 1]'),
  (41, 'first_3day_events', '趋势衰减', '入学前 3 天事件数（冷启动信号）'),
  (42, 'last_3day_events', '趋势衰减', '窗口最后 3 天事件数'),
  (43, 'course_id', '类别标识', '课程 ID（CatBoost 原生类别特征）'),
  (44, 'username', '类别标识', '学习者 ID（CatBoost 原生类别特征，高基数）');

-- -----------------------------------------------------------------------------
-- 5. 特征向量表：一行 = 一条选课记录的 44 维特征（宽表，列序与 feature_dict 一致）
--    实现中不落盘：特征由 FeatureBuilder 在训练与推理时实时构造；data/processed/ 只有派生的标签表。
--    因样本请求只有 6 条事件，按 14 天窗构造出的向量属演示性质，故不写入行数据。
-- -----------------------------------------------------------------------------
CREATE TABLE feature_vector (
  enrollment_id VARCHAR(32) NOT NULL COMMENT '选课记录，一行一条',
  window_days   SMALLINT    NOT NULL DEFAULT 14 COMMENT '构造该行时使用的观察窗长度',
  total_events                 DOUBLE COMMENT '观察窗内行为事件总数',
  active_days                  DOUBLE COMMENT '有行为记录的 distinct 天数',
  unique_objects               DOUBLE COMMENT '触及的 distinct 学习对象数',
  span_days                    DOUBLE COMMENT '首个行为到最后一个行为的天数跨度',
  days_since_last_event        DOUBLE COMMENT '距观察窗结束的天数（越大说明越早沉默）',
  events_per_active_day        DOUBLE COMMENT '活跃日均事件数',
  events_per_day               DOUBLE COMMENT '按窗口长度折算的日均事件数',
  active_day_ratio             DOUBLE COMMENT '活跃天数占观察窗比例',
  span_ratio                   DOUBLE COMMENT '行为跨度占观察窗比例',
  n_video                      DOUBLE COMMENT '视频类事件数',
  n_problem                    DOUBLE COMMENT '习题类事件数',
  n_discussion                 DOUBLE COMMENT '论坛类事件数',
  n_navigate                   DOUBLE COMMENT '页面导航事件数',
  n_page_close                 DOUBLE COMMENT '页面关闭事件数',
  n_access                     DOUBLE COMMENT '课程访问事件数',
  n_wiki                       DOUBLE COMMENT 'wiki 浏览事件数',
  video_ratio                  DOUBLE COMMENT '视频事件占全部事件比例',
  problem_ratio                DOUBLE COMMENT '习题事件占比',
  discussion_ratio             DOUBLE COMMENT '论坛事件占比',
  navigate_ratio               DOUBLE COMMENT '导航事件占比',
  video_unique_objects         DOUBLE COMMENT '观看过的 distinct 视频数',
  video_replay_ratio           DOUBLE COMMENT '视频播放次数 / 去重视频数（重复观看强度）',
  problem_unique_objects       DOUBLE COMMENT '作答过的 distinct 题目数',
  problem_repeat_ratio         DOUBLE COMMENT '习题提交次数 / 去重题目数（反复尝试强度）',
  discussion_unique_objects    DOUBLE COMMENT '参与的 distinct 讨论区/帖子数',
  night_event_ratio            DOUBLE COMMENT '22:00-06:00 事件占比',
  weekend_event_ratio          DOUBLE COMMENT '周末事件占比',
  hour_entropy                 DOUBLE COMMENT '行为小时分布的香农熵，衡量学习节律规律性',
  n_sessions                   DOUBLE COMMENT '按 30 分钟间隔切分的学习会话数',
  mean_session_events          DOUBLE COMMENT '平均每个会话的事件数',
  max_session_events           DOUBLE COMMENT '最长一次会话的事件数',
  mean_gap_hours               DOUBLE COMMENT '相邻事件平均间隔（小时）',
  gap_std_hours                DOUBLE COMMENT '相邻事件间隔标准差（小时）',
  max_gap_days                 DOUBLE COMMENT '最长一次中断的天数',
  decayed_events               DOUBLE COMMENT '按 exp(-λ·剩余天数) 加权后的事件量',
  decayed_active_days          DOUBLE COMMENT '按时间衰减加权后的活跃天数',
  early_events                 DOUBLE COMMENT '窗口前半段事件数',
  late_events                  DOUBLE COMMENT '窗口后半段事件数',
  late_event_ratio             DOUBLE COMMENT '后半段事件占比，衡量学习持续性',
  activity_trend               DOUBLE COMMENT '(后半 - 前半) / (后半 + 前半)，取值 [-1, 1]',
  first_3day_events            DOUBLE COMMENT '入学前 3 天事件数（冷启动信号）',
  last_3day_events             DOUBLE COMMENT '窗口最后 3 天事件数',
  course_id                    DOUBLE COMMENT '课程 ID（CatBoost 原生类别特征）',
  username                     DOUBLE COMMENT '学习者 ID（CatBoost 原生类别特征，高基数）',
  PRIMARY KEY (enrollment_id),
  CONSTRAINT fk_vec_enroll FOREIGN KEY (enrollment_id) REFERENCES enrollment (enrollment_id)
) ENGINE=InnoDB COMMENT='44 维特征宽表';

-- -----------------------------------------------------------------------------
-- 6. 模型产物表：一行 = 一个可部署的模型版本，对应 artifacts/<run>/model/ 目录
--    阈值与指标取自 metadata.json，可逐项复核
-- -----------------------------------------------------------------------------
CREATE TABLE model_artifact (
  model_version    VARCHAR(32)  NOT NULL COMMENT '版本号（UTC 时间戳），metadata.json 的 model_version',
  trained_at       DATETIME     NOT NULL COMMENT '训练完成时间',
  model_name       VARCHAR(24)  NOT NULL COMMENT '模型族，本项目为 catboost',
  threshold        DECIMAL(6,4) NOT NULL COMMENT '决策阈值（训练集 OOF 上精确搜索得到）',
  threshold_strategy VARCHAR(24) NOT NULL COMMENT '阈值选取策略，本项目为 oof_cost',
  cost_fn          DECIMAL(8,2) NOT NULL COMMENT '漏判代价 C_FN',
  cost_fp          DECIMAL(8,2) NOT NULL COMMENT '误判代价 C_FP',
  cost_ratio       DECIMAL(6,2) NOT NULL COMMENT '代价比 C_FN / C_FP',
  n_train          INT          NOT NULL COMMENT '训练集样本数',
  n_test           INT          NOT NULL COMMENT '测试集样本数',
  roc_auc          DECIMAL(6,4) NULL COMMENT '测试集 ROC-AUC',
  pr_auc           DECIMAL(6,4) NULL COMMENT '测试集 PR-AUC',
  recall           DECIMAL(6,4) NULL COMMENT '测试集召回率（按本阈值）',
  cost_per_student DECIMAL(8,2) NULL COMMENT '测试集人均期望代价（元）',
  artifact_path    VARCHAR(200) NOT NULL COMMENT '产物目录，相对仓库根',
  PRIMARY KEY (model_version)
) ENGINE=InnoDB COMMENT='模型版本与产物';

INSERT INTO model_artifact (model_version, trained_at, model_name, threshold, threshold_strategy,
  cost_fn, cost_fp, cost_ratio, n_train, n_test, roc_auc, pr_auc, recall, cost_per_student, artifact_path) VALUES
  ('20261023T124525Z', '2026-10-23 12:45:25', 'catboost', 0.1048, 'oof_cost', 1000.00, 100.00, 10.00, 96608, 23934, 0.8407, 0.9378, 0.9988, 20.54, 'artifacts/realdata_official/model');

-- -----------------------------------------------------------------------------
-- 7. 预测记录表：每次 POST /v1/predict 落一行；实现中不持久化，这里给出表结构
--    并写入一条真实预测（对 data/sample/request.json 的实测响应）
-- -----------------------------------------------------------------------------
CREATE TABLE prediction_log (
  prediction_id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '预测流水号',
  enrollment_id     VARCHAR(32)  NOT NULL COMMENT '被预测的选课记录',
  model_version     VARCHAR(32)  NOT NULL COMMENT '使用的模型版本',
  risk_probability  DECIMAL(6,4) NOT NULL COMMENT '辍学概率（已校准）',
  risk_level        ENUM('high','medium','low') NOT NULL COMMENT '风险等级',
  decision_threshold DECIMAL(6,4) NOT NULL COMMENT '本次决策使用的阈值',
  top_factor        VARCHAR(48)  NULL COMMENT '贡献最大的因子名',
  top_factor_value  DECIMAL(10,4) NULL COMMENT '该因子的贡献（log-odds 增量）',
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (prediction_id),
  KEY idx_pred_enroll (enrollment_id),
  CONSTRAINT fk_pred_enroll FOREIGN KEY (enrollment_id) REFERENCES enrollment (enrollment_id),
  CONSTRAINT fk_pred_model  FOREIGN KEY (model_version) REFERENCES model_artifact (model_version)
) ENGINE=InnoDB COMMENT='预测记录';

-- -----------------------------------------------------------------------------
-- 8. 干预名单表：助教派单用，按 risk_probability 降序即可复现 GET/POST /v1/intervention-list
-- -----------------------------------------------------------------------------
CREATE TABLE intervention_list (
  list_id        BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '名单行号，即导出 CSV 的 priority',
  prediction_id  BIGINT UNSIGNED NOT NULL COMMENT '来源预测记录',
  assigned_to    VARCHAR(64)  NULL COMMENT '负责跟进的助教',
  status         ENUM('pending','contacted','closed','ignored') NOT NULL DEFAULT 'pending',
  created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (list_id),
  KEY idx_list_pred (prediction_id),
  CONSTRAINT fk_list_pred FOREIGN KEY (prediction_id) REFERENCES prediction_log (prediction_id)
) ENGINE=InnoDB COMMENT='干预名单';

-- -----------------------------------------------------------------------------
-- 9. 样本数据：1 门课程 + 1 条选课 + 6 条行为事件（取自 data/sample/request.json）
-- -----------------------------------------------------------------------------
INSERT INTO course (course_id, course_name, start_date, window_days) VALUES
  ('C01', '演示课程 C01（MOOC 辍学预警）', '2014-03-01', 14);

INSERT INTO enrollment (enrollment_id, course_id, username, label_dropout, split_name) VALUES
  ('E9000001', 'C01', 'u000001', NULL, NULL);

INSERT INTO behavior_event (enrollment_id, event_time, source, event, object) VALUES
  ('E9000001', '2014-03-01 19:00:00', 'video', 'play_video', 'C01_video_01'),
  ('E9000001', '2014-03-01 19:24:00', 'video', 'pause_video', 'C01_video_01'),
  ('E9000001', '2014-03-02 10:12:00', 'problem', 'open', 'C01_problem_01'),
  ('E9000001', '2014-03-02 10:35:00', 'problem', 'submit', 'C01_problem_01'),
  ('E9000001', '2014-03-03 21:05:00', 'discussion', 'create_post', 'C01_forum_01'),
  ('E9000001', '2014-03-05 08:40:00', 'access', 'page_close', 'C01_page_01');

-- 一条真实预测记录：对上述请求调用 POST /v1/predict 的实测响应
INSERT INTO prediction_log (enrollment_id, model_version, risk_probability, risk_level,
  decision_threshold, top_factor, top_factor_value) VALUES
  ('E9000001', '20261023T124525Z', 0.8398, 'high', 0.1048, 'problem_ratio', -0.1875);

-- 名单行由预测记录生成，按 risk_probability 降序取前 N 条：
-- INSERT INTO intervention_list (prediction_id, assigned_to, status)
--   SELECT prediction_id, '助教A', 'pending' FROM prediction_log
--    WHERE risk_probability >= decision_threshold ORDER BY risk_probability DESC;

SET FOREIGN_KEY_CHECKS = 1;

-- -----------------------------------------------------------------------------
-- 复核用查询（导入后可直接执行，结果应与交付文档一致）
-- -----------------------------------------------------------------------------
-- ① 44 维特征按分组计数：
--    SELECT f_group, COUNT(*) FROM feature_dict GROUP BY f_group ORDER BY 2 DESC;
-- ② 当前部署模型与阈值：
--    SELECT model_version, threshold, cost_fn, cost_fp, roc_auc, cost_per_student FROM model_artifact;
-- ③ 本次预测的风险等级与主要因子：
--    SELECT enrollment_id, risk_probability, risk_level, top_factor, top_factor_value
--      FROM prediction_log ORDER BY created_at DESC LIMIT 5;
-- ④ 干预名单（按风险降序，对应导出 CSV）：
--    SELECT p.enrollment_id, p.risk_probability, p.risk_level, p.top_factor
--      FROM prediction_log p
--     WHERE p.risk_probability >= p.decision_threshold
--     ORDER BY p.risk_probability DESC;
-- =============================================================================
-- 文件结束
