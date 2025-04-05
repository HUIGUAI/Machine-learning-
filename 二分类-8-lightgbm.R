# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---lightgbm

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/boost_tree.html
# https://parsnip.tidymodels.org/reference/details_boost_tree_lightgbm.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html


library(tidymodels)
library(bonsai)

# 读取数据
Heart <- readr::read_csv(file.choose()) # tibble
colnames(Heart)

# 修正变量类型
# 将分类变量转换为factor
for(i in c(3,4,7,8,10,12,14,15)){
  Heart[[i]] <- factor(Heart[[i]])
}
# 变量类型修正后数据概况
skimr::skim(Heart)

###############################################################

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)

###############################################################

# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(AHD ~ ., traindata) %>%
  step_rm(Id) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  dplyr::select(AHD, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  dplyr::select(AHD, everything())

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

###############################################################

# 训练模型
# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_model(model_lightgbm) %>%
  add_formula(AHD ~ .)
wk_lightgbm

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# 超参数寻优范围
hpset_lightgbm <- parameters(
  tree_depth(range = c(1, 3)),
  trees(range = c(100, 500)),
  learn_rate(range = c(-3, -1)),
  mtry(range = c(2, 8)),
  min_n(range = c(5, 10)),
  loss_reduction(range = c(-3, 0))
)
# hpgrid_lightgbm <- grid_regular(hpset_lightgbm, levels = 2)
set.seed(42)
hpgrid_lightgbm <- grid_random(hpset_lightgbm, size = 5)
hpgrid_lightgbm

# 交叉验证随机搜索过程
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_grid(resamples = folds,
            grid = hpgrid_lightgbm,
            metrics = metric_set(yardstick::accuracy, 
                                 yardstick::roc_auc, 
                                 yardstick::pr_auc),
            control = control_grid(save_pred = T, verbose = T))

##################################################
# 贝叶斯优化超参数可以取代前面的网格搜索
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    param_info = extract_parameter_set_dials(wk_lightgbm) %>%
      update(mtry = finalize(mtry(), traindata2[,-1])),
    initial = 10,
    iter = 30,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, verbose = T, no_improve = 5)
  )
##################################################

# 图示交叉验证结果
autoplot(tune_lightgbm)
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_best(metric = "roc_auc")
hpbest_lightgbm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(traindata2)
final_lightgbm

# 提取最终的算法模型
final_lightgbm2 <- final_lightgbm %>%
  extract_fit_engine()

# 变量重要性
lgb.importance(final_lightgbm2, percentage = T)
lgb.plot.importance(lgb.importance(final_lightgbm2, percentage = T))

# 变量对预测的贡献
lgb.interprete(final_lightgbm2, 
               as.matrix(testdata2[,-1]), 
               1:2)[[2]]
lgb.plot.interpretation(
  lgb.interprete(final_lightgbm2, as.matrix(testdata2[,-1]), 2)[[1]]
)


###############################################################

# 应用模型-预测训练集
predtrain_lightgbm <- final_lightgbm %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(AHD)) %>%
  mutate(dataset = "train")
predtrain_lightgbm 
# 评估模型ROC曲线-训练集上
levels(traindata2$AHD)
roctrain_lightgbm <- predtrain_lightgbm %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
roctrain_lightgbm
autoplot(roctrain_lightgbm)

# 约登法则对应的p值
yueden_lightgbm <- roctrain_lightgbm %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_lightgbm
# 预测概率+约登法则=预测分类
predtrain_lightgbm2 <- predtrain_lightgbm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_lightgbm, "No", "Yes")))
predtrain_lightgbm2
# 混淆矩阵
cmtrain_lightgbm <- predtrain_lightgbm2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtrain_lightgbm
autoplot(cmtrain_lightgbm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_lightgbm <- cmtrain_lightgbm %>%
  summary(event_level = "first") %>%
  bind_rows(predtrain_lightgbm %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "train")
eval_train_lightgbm

###############################################################

# 应用模型-预测测试集
predtest_lightgbm <- final_lightgbm %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(AHD)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "lightgbm")
predtest_lightgbm

# 评估模型ROC曲线-测试集上
roctest_lightgbm <- predtest_lightgbm %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  mutate(dataset = "test")
roctest_lightgbm
autoplot(roctest_lightgbm)


# 预测概率+约登法则=预测分类
predtest_lightgbm2 <- predtest_lightgbm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_lightgbm, "No", "Yes")))
predtest_lightgbm2
# 混淆矩阵
cmtest_lightgbm <- predtest_lightgbm2 %>%
  conf_mat(truth = AHD, estimate = .pred_class)
cmtest_lightgbm
autoplot(cmtest_lightgbm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_lightgbm <- cmtest_lightgbm %>%
  summary(event_level = "first") %>%
  bind_rows(predtest_lightgbm %>%
              roc_auc(AHD, .pred_No, event_level = "first")) %>%
  mutate(dataset = "test")
eval_test_lightgbm

###############################################################


# 合并训练集和测试集上ROC曲线
roctrain_lightgbm %>%
  bind_rows(roctest_lightgbm) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_lightgbm <- eval_train_lightgbm %>%
  bind_rows(eval_test_lightgbm) %>%
  mutate(model = "lightgbm")
eval_lightgbm

#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_lightgbm <- eval_tune_lightgbm %>%
  inner_join(hpbest_lightgbm[, 1:6])
eval_best_cv_lightgbm

# 最优超参数的交叉验证指标具体结果
eval_best_cv5_lightgbm <- tune_lightgbm %>%
  collect_predictions() %>%
  inner_join(hpbest_lightgbm[, 1:6]) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_No) %>%
  ungroup() %>%
  mutate(model = "lightgbm") %>%
  inner_join(eval_best_cv_lightgbm[c(7,9,11)])
eval_best_cv5_lightgbm

# 保存评估结果
save(final_lightgbm,
     predtest_lightgbm,
     eval_lightgbm,
     eval_best_cv5_lightgbm, 
     file = ".\\cls2\\evalresult_lightgbm.RData")

# 保存lightgbm模型比较特殊
model_file <- 
  tempfile(pattern = "lightgbm", tmpdir = ".", fileext = ".txt")
lightgbm::lgb.save(final_lightgbm2, model_file)

# # 加载也需要自己的函数
# load_booster <- lgb.load(file.choose())

# 最优超参数的交叉验证指标图示
eval_best_cv5_lightgbm %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_lightgbm %>%
  collect_predictions() %>%
  inner_join(hpbest_lightgbm[, 1:6]) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_No, event_level = "first") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(linewidth = 1) +
  theme_bw()


###################################################################

# 自变量数据集
colnames(traindata2)
traindatax <- traindata2[,-1]
colnames(traindatax)

# iml包
library(iml)
predictor_model <- Predictor$new(
  final_lightgbm, 
  data = traindatax,
  y = traindata2$AHD,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      pull(2)
  }
)

# 变量重要性-基于置换
imp_model <- FeatureImp$new(
  predictor_model, 
  loss = function(actual, predicted){
    return(1-Metrics::auc(as.numeric(actual=="Yes"), predicted))
  }
)
# 数值
imp_model$results
# 图示
imp_model$plot() +
  theme_bw()


# 变量效应
pdp_model <- FeatureEffect$new(
  predictor_model, 
  feature = "Age",
  method = "pdp"
)
# 数值
pdp_model$results
# 图示
pdp_model$plot() +
  theme_bw()

# 所有变量的效应全部输出
effs_model <- FeatureEffects$new(predictor_model, method = "pdp")
# 数值
effs_model$results
# 图示
effs_model$plot()

# 单样本shap分析
shap_model <- Shapley$new(
  predictor_model, 
  x.interest = traindatax[1,]
)
# 数值
shap_model$results
# 图示
shap_model$plot() +
  theme_bw()

# 基于所有样本的shap分析
# fastshap包
library(fastshap)
shap <- explain(
  final_lightgbm, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata, type = "prob") %>% pull(2)
  }
)

# 单样本图示
force_plot(object = shap[1L, ], 
           feature_values = as.data.frame(traindatax)[1L, ], 
           baseline = mean(predtrain_lightgbm$.pred_Yes), 
           display = "viewer") 

# 变量重要性
autoplot(shap, fill = "skyblue") +
  theme_bw()

data1 <- shap %>%
  as.data.frame() %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1), values_to = "shap")
shapimp <- data1 %>%
  dplyr::group_by(name) %>%
  dplyr::summarise(shap.abs.mean = mean(abs(shap))) %>%
  dplyr::arrange(shap.abs.mean) %>%
  dplyr::mutate(name = forcats::as_factor(name))
data2 <- traindatax  %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1))

# 所有变量shap图示
library(ggbeeswarm)
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(
    value = (value - min(value)) / (max(value) - min(value)),
    feature = factor(feature, levels = levels(shapimp$name))
  ) %>%
  dplyr::arrange(value) %>%
  dplyr::ungroup() %>%
  ggplot(aes(x = shap, y = feature, color = value)) +
  geom_quasirandom(width = 0.2) +
  scale_color_gradient(
    low = "red", 
    high = "blue", 
    breaks = c(0, 1), 
    labels = c(" Low", "High "), 
    guide = guide_colorbar(barwidth = 1, 
                           barheight = 20,
                           ticks = F,
                           title.position = "right",
                           title.hjust = 0.5)
  ) +
  labs(x = "SHAP value", color = "Feature value") +
  theme_bw() +
  theme(legend.title = element_text(angle = -90))



# 单变量shap图示
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::filter(feature == "MaxHR") %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point() +
  geom_smooth(se = F, span = 0.5) +
  labs(x = "MaxHR") +
  theme_bw()










