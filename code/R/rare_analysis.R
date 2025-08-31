library(tidyr)
library(broom)

# 读取数据
df <- read.csv("../data/rare_invade_hpc.csv")

pal_rgb <- c("1" = "#E74C3C", "2" = "#2ECC71", "3" = "#3498DB")

base_theme <- theme_minimal(base_size = 14) +
  theme(
    text       = element_text(family = "Times New Roman"),
    axis.text  = element_text(family = "Times New Roman", size = 14),
    axis.title = element_text(family = "Times New Roman", size = 14),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks       = element_line(color = "black", size = 0.3),
    axis.ticks.length = unit(0.15, "cm")
  )


# 生存标签
df <- df %>%
  mutate(survival     = ifelse(C_final > 1e-5, "Survived", "Extinct"),
         survived_bin = ifelse(survival == "Survived", 1, 0))

df_surv <- df %>% filter(survival == "Survived")

# 1) CUE 对生存概率
p_s <-ggplot(df, aes(x = CUE, y = survived_bin, colour = factor(DilutionRate))) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.3) +
  stat_smooth(method = "glm",
              method.args = list(family = "binomial"),
              se = FALSE) +
  labs(x = "CUE",
       y = "Probability of Survival",
       colour = "Dilution Rate") +
  base_theme          
p_s
glm(survived_bin ~ CUE * DilutionRate, data = df, family = "binomial") %>%
  tidy()

ggsave("../results/survival_by_dilution.pdf",
       plot = p_s,
       device = cairo_pdf,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("/home/jiayi-chen/Documents/presentation/survival_by_dilution.png",
       plot = p_s,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

# 2) 生存个体 C_final ~ CUE
p_a <- ggplot(df_surv, aes(x = CUE, y = C_final, colour = factor(DilutionRate))) +
  geom_point(alpha = 0.5) +
  scale_y_log10() +
  labs(x = "CUE",
       y = "Abundance",
       colour = "Dilution Rate") +
  base_theme          

model <- glm(log10(C_final) ~ CUE * DilutionRate, data = df_surv)
summary(model)

p_a
ggsave("../results/rare_abundance_CUE.pdf",
       plot = p_a,
       device = cairo_pdf,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")

ggsave("/home/jiayi-chen/Documents/presentation/rare_abundance_CUE.png",
       plot = p_a,
       width = 21,
       height = 10,
       units = "cm",
       dpi = 600,
       bg = "white")
