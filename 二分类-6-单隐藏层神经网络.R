# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---单隐藏层神经网络

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/mlp.html
# https://parsnip.tidymodels.org/reference/details_mlp_nnet.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

library(tidymodels)

# 读取数据
Heart <- readr::read_csv(file.choose())
colnames(Heart)
TP <- readr::read_csv(file.choose("D:/研一/研一下/利奈唑胺/机器学习new变量/newml.csv"))
colnames(TP)
EX <- readr::read_csv(file.choose("D:/研一/研一下/利奈唑胺/机器学习new变量/test72.csv"))
colnames(EX)

# 修正变量类型
# 将分类变量转换为factor
for(i in c(3,4,7,8,10,12,14,15)){
  Heart[[i]] <- factor(Heart[[i]])
}
for(i in c(1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,19,20,21,22,23,24)){
  TP[[i]] <- factor(TP[[i]])
}
for(i in c(1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,19,20,21,22,23,24)){
  EX[[i]] <- factor(EX[[i]])
}
# factor(Heart[,4])
# 变量类型修正后数据概况
skimr::skim(Heart)
skimr::skim(TP)
skimr::skim(EX)

#############################################################

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)
traindata <- TP
testdata <- EX
#############################################################

# 数据预处理
# 先对照训练集写配方
# recipes包
datarecipe <- recipe(status ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  dplyr::select(status, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  dplyr::select(status, everything())

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

#############################################################

# 训练模型
# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_args(MaxNWts = 5000)
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_model(model_mlp) %>%
  add_formula(status ~ .)
wk_mlp

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# 超参数寻优范围
hpset_mlp <- parameters(hidden_units(range = c(15, 24)),
                    penalty(range = c(-3, 0)),
                    epochs(range = c(50, 150)))
hpgrid_mlp <- grid_regular(hpset_mlp, levels = 2)
hpgrid_mlp


# 交叉验证网格搜索过程
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_grid(resamples = folds,
            grid = hpgrid_mlp,
            metrics = metric_set(yardstick::accuracy, 
                                 yardstick::roc_auc, 
                                 yardstick::pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_mlp)
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_best(metric = "roc_auc")
hpbest_mlp

# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata2)
final_mlp

# 提取最终的算法模型
final_mlp2 <- final_mlp %>%
  extract_fit_engine()

library(NeuralNetTools)
plotnet(final_mlp2)
garson(final_mlp2) +
  coord_flip()
olden(final_mlp2) +
  coord_flip()

#############################################################

# 应用模型-预测训练集
predtrain_mlp <- final_mlp %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(status)) %>%
  mutate(dataset = "train")
predtrain_mlp
# 评估模型ROC曲线-训练集上
levels(traindata2$status)
roctrain_mlp <- predtrain_mlp %>%
  roc_curve(status, .pred_0, event_level = "first") %>%
  mutate(dataset = "train")
roctrain_mlp
autoplot(roctrain_mlp)

# 约登法则对应的p值
yueden_mlp <- roctrain_mlp %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_mlp
# 预测概率+约登法则=预测分类
predtrain_mlp2 <- predtrain_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_0 >= yueden_mlp, "0", "1")))
predtrain_mlp2
# 混淆矩阵
cmtrain_mlp <- predtrain_mlp2 %>%
  conf_mat(truth = status, estimate = .pred_class)
cmtrain_mlp
autoplot(cmtrain_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_mlp <- cmtrain_mlp %>%
  summary(event_level = "first") %>%
  bind_rows(predtrain_mlp %>%
              roc_auc(status, .pred_0, event_level = "first")) %>%
  mutate(dataset = "train")
eval_train_mlp

#############################################################

# 应用模型-预测测试集
predtest_mlp <- final_mlp %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(status)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "mlp")
predtest_mlp
# 评估模型ROC曲线-测试集上
roctest_mlp <- predtest_mlp %>%
  roc_curve(status, .pred_0, event_level = "first") %>%
  mutate(dataset = "test")
roctest_mlp
autoplot(roctest_mlp)


# 预测概率+约登法则=预测分类
predtest_mlp2 <- predtest_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_0 >= yueden_mlp, "0", "1")))
predtest_mlp2
# 混淆矩阵
cmtest_mlp <- predtest_mlp2 %>%
  conf_mat(truth = status, estimate = .pred_class)
cmtest_mlp
autoplot(cmtest_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_mlp <- cmtest_mlp %>%
  summary(event_level = "first") %>%
  bind_rows(predtest_mlp %>%
              roc_auc(status, .pred_0, event_level = "first")) %>%
  mutate(dataset = "test")
eval_test_mlp

#############################################################

# 合并训练集和测试集上ROC曲线
roctrain_mlp %>%
  bind_rows(roctest_mlp) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  theme_bw()


# 合并训练集和测试集上性能指标
eval_mlp <- eval_train_mlp %>%
  bind_rows(eval_test_mlp) %>%
  mutate(model = "mlp")
eval_mlp

#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_mlp <- eval_tune_mlp %>%
  inner_join(hpbest_mlp[, 1:3])
eval_best_cv_mlp

# 最优超参数的交叉验证指标具体结果
eval_best_cv5_mlp <- tune_mlp %>%
  collect_predictions() %>%
  inner_join(hpbest_mlp[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_No) %>%
  ungroup() %>%
  mutate(model = "mlp") %>%
  inner_join(eval_best_cv_mlp[c(4,6,8)])
eval_best_cv5_mlp

# 保存评估结果
save(final_mlp,
     predtest_mlp,
     eval_mlp,
     eval_best_cv5_mlp, 
     file = ".\\cls2\\evalresult_mlp.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv5_mlp %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_mlp %>%
  collect_predictions() %>%
  inner_join(hpbest_mlp[, 1:3]) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_No) %>%
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
  final_mlp, 
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
  final_mlp, 
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
           baseline = mean(predtrain_mlp$.pred_Yes), 
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









