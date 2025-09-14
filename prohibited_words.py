import nonebot
from nonebot.adapters.onebot.v11 import (
    Bot,
    MessageEvent,
    GroupMessageEvent,
    Message,
    GROUP_ADMIN,
    GROUP_OWNER
)
from nonebot.plugin import PluginMetadata, on_command
from nonebot.permission import SUPERUSER
from nonebot.params import CommandArg
import sqlite3
import jieba
import jieba.analyse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, DefaultDict
from collections import defaultdict
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from nonebot import get_driver

# 导入主插件的核心功能
from .main_plugin import config, enhanced_ai

__plugin_meta__ = PluginMetadata(
    name="智能学习优化助手",
    description="提供学习优化功能，提升机器人学习效率与智能度",
    usage="命令管理学习优化设置",
    extra={"author": "YourName", "version": "2.0.2"}  # 更新版本号
)

# 获取主插件的日志器
logger = config.logger

# ========== 学习优化设置管理 ==========
class OptimizationSettingsManager:
    def __init__(self):
        # 学习优化设置文件路径
        self.optimization_path = config.data_path / "optimization_settings.json"
        # 学习优化设置 {group_id: setting_dict}
        self.learning_optimization: DefaultDict[int, Dict[str, bool]] = defaultdict(dict)
        # 确保目录存在
        config.data_path.mkdir(parents=True, exist_ok=True)
        # 加载配置
        self.load_settings()

    def load_settings(self):
        """从本地JSON文件加载学习优化设置"""
        try:
            if self.optimization_path.exists():
                with open(self.optimization_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 转换键为整数
                    for group_id, settings in data.items():
                        self.learning_optimization[int(group_id)] = settings
                logger.info("学习优化设置加载完成")
            else:
                # 文件不存在则创建空文件
                with open(self.optimization_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                logger.info("创建新的学习优化设置文件")
        except Exception as e:
            logger.error(f"加载学习优化设置失败: {e}")

    def save_settings(self):
        """保存学习优化设置到本地JSON文件"""
        try:
            # 转换为字符串键的字典
            save_data = {str(k): v for k, v in self.learning_optimization.items()}
            with open(self.optimization_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logger.info("学习优化设置已保存")
        except Exception as e:
            logger.error(f"保存学习优化设置失败: {e}")

    def set_optimization_setting(self, group_id: int, setting_name: str, value: bool):
        """设置学习优化选项"""
        if group_id not in self.learning_optimization:
            self.learning_optimization[group_id] = {}
        self.learning_optimization[group_id][setting_name] = value
        self.save_settings()

    def get_optimization_setting(self, group_id: int, setting_name: str, default: bool) -> bool:
        """获取学习优化设置"""
        group_settings = self.learning_optimization.get(group_id, {})
        return group_settings.get(setting_name, default)

# 初始化学习优化设置管理器
settings_manager = OptimizationSettingsManager()

# ========== 高效学习优化系统 ==========
class LearningOptimizer:
    def __init__(self):
        # 词性过滤列表
        self.allowed_pos_tags = {"n", "v", "a", "vn", "an", "ad"}
        # 高级语义分析模型
        self.advanced_vectorizer = None
        # 缓存TF-IDF向量
        self.message_vectors_cache = None
        self.message_ids_cache = None
        self.messages_cache = None
        self.last_update_time = 0
        # 缓存最后训练时间
        self.last_trained_time = 0

    async def preload_message_vectors(self):
        """预加载消息向量到内存"""
        try:
            # 获取所有消息
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, message FROM chat_logs")
                results = cursor.fetchall()
                if not results:
                    self.message_vectors_cache = None
                    self.message_ids_cache = None
                    self.messages_cache = None
                    logger.info("没有聊天记录可供预加载")
                    return
                
                self.messages_cache = [row[1] for row in results]
                self.message_ids_cache = [row[0] for row in results]
                
                # 使用缓存向量化器
                vectorizer = await self.get_vectorizer()
                if vectorizer:
                    self.message_vectors_cache = vectorizer.transform(self.messages_cache)
                    self.last_update_time = time.time()
                    logger.info(f"预加载{len(self.messages_cache)}条消息向量到内存")
        except Exception as e:
            logger.error(f"预加载消息向量失败: {e}")

    async def get_vectorizer(self):
        """获取向量化器（优先使用高级模型）"""
        try:
            # 如果高级模型不存在或过期（超过1天）
            if (not self.advanced_vectorizer or 
                time.time() - self.last_trained_time > 86400):
                await self.train_advanced_model()
            
            # 如果主插件向量化器未初始化
            if not enhanced_ai.vectorizer:
                await enhanced_ai.train_vectorizer()
            
            # 优先使用高级模型
            if self.advanced_vectorizer:
                return self.advanced_vectorizer
            return enhanced_ai.vectorizer
        except Exception as e:
            logger.error(f"获取向量化器失败: {e}")
            return None

    def enhanced_extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """高性能关键词提取 - 使用词性过滤和语义分析"""
        try:
            # 使用jieba提取带权重的关键词
            keywords_with_weights = jieba.analyse.extract_tags(
                text, 
                topK=top_n * 2,
                withWeight=True,
                allowPOS=tuple(self.allowed_pos_tags)
            )
            # 过滤停用词和短词
            filtered_keywords = []
            for word, weight in keywords_with_weights:
                if (word not in config.stop_words and 
                    len(word) >= config.min_keyword_length):
                    filtered_keywords.append((word, weight))
            # 按权重排序并截取
            filtered_keywords.sort(key=lambda x: x[1], reverse=True)
            return [word for word, _ in filtered_keywords[:top_n]]
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []

    async def enhanced_find_relevant(self, query: str, context: List[str] = None) -> Optional[str]:
        """高性能相似度匹配 - 结合语义和上下文分析"""
        try:
            # 获取向量化器
            vectorizer = await self.get_vectorizer()
            if not vectorizer:
                logger.warning("向量化器不可用，无法进行相似度匹配")
                return None
                
            # 每5分钟检查一次缓存更新
            if not self.message_vectors_cache or time.time() - self.last_update_time > 300:
                await self.preload_message_vectors()
                
            # 使用缓存向量
            if self.message_vectors_cache is None:
                logger.warning("消息向量缓存为空，无法进行相似度匹配")
                return None
                
            # 组合查询和上下文
            full_query = " ".join(context) + " " + query if context else query
            
            # 向量化查询
            query_vec = vectorizer.transform([full_query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vec, self.message_vectors_cache)[0]
            
            # 上下文增强
            if context and settings_manager.get_optimization_setting(0, "context_enhancement", True):
                context_keywords = self.enhanced_extract_keywords(" ".join(context))
                if context_keywords:
                    context_query = " ".join(context_keywords)
                    context_vec = vectorizer.transform([context_query])
                    context_similarities = cosine_similarity(context_vec, self.message_vectors_cache)[0]
                    # 加权合并相似度
                    similarities = (similarities * 0.7) + (context_similarities * 0.3)
            
            # 多样性选择
            top_indices = np.argsort(similarities)[-10:]
            top_similarities = similarities[top_indices]
            exp_scores = np.exp(top_similarities * 3)  # 温度参数控制多样性
            probs = exp_scores / np.sum(exp_scores)
            selected_idx = np.random.choice(top_indices, p=probs)
            
            # 检查相似度阈值
            if similarities[selected_idx] < config.min_similarity:
                logger.debug(f"相似度 {similarities[selected_idx]:.2f} 低于阈值 {config.min_similarity}")
                return None
                
            # 直接从缓存获取消息
            return self.messages_cache[selected_idx]
                
        except Exception as e:
            logger.error(f"相似度匹配失败: {e}")
            return None

    async def train_advanced_model(self):
        """训练高级语义分析模型"""
        try:
            # 获取所有消息
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT message FROM chat_logs")
                results = cursor.fetchall()
                messages = [row[0] for row in results] if results else []
                
            if len(messages) < 50:
                logger.warning(f"消息数量不足({len(messages)})，跳过高级模型训练")
                return
                
            logger.info(f"开始训练高级语义模型，样本数: {len(messages)}")
            
            # 创建高级TF-IDF向量化器
            self.advanced_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=list(config.stop_words)
            )
            self.advanced_vectorizer.fit(messages)
            
            # 更新缓存
            await self.preload_message_vectors()
            self.last_trained_time = time.time()
            logger.info(f"高级语义模型训练完成，特征数: {len(self.advanced_vectorizer.get_feature_names_out())}")
        except Exception as e:
            logger.error(f"训练高级模型失败: {e}")

# 初始化学习优化器
learning_optimizer = LearningOptimizer()

# ========== 学习优化设置命令 ==========
optimization_setting = on_command("学习优化设置", permission=GROUP_ADMIN | GROUP_OWNER | SUPERUSER)
@optimization_setting.handle()
async def handle_optimization_setting(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """设置学习优化选项"""
    arg_str = args.extract_plain_text().strip()
    group_id = event.group_id
    
    if not arg_str:
        # 显示当前设置
        settings = settings_manager.learning_optimization.get(group_id, {})
        msg = "学习优化设置:\n"
        msg += f"高级语义匹配: {settings.get('advanced_semantic', True)}\n"
        msg += f"上下文增强: {settings.get('context_enhancement', True)}\n"
        msg += f"实时向量缓存: {settings.get('vector_caching', True)}"
        await optimization_setting.finish(msg)
        
    try:
        # 解析设置参数
        parts = arg_str.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError("格式错误，请使用: /学习优化设置 [选项] [开/关]")
            
        option, value = parts
        value_map = {"开": True, "关": False, "on": True, "off": False, "true": True, "false": False}
        if value.lower() not in value_map:
            raise ValueError(f"值必须是'开'或'关'，当前值: {value}")
            
        # 有效选项列表
        valid_options = {
            "高级语义匹配": "advanced_semantic",
            "上下文增强": "context_enhancement",
            "实时向量缓存": "vector_caching"
        }
        
        if option not in valid_options:
            raise ValueError(f"无效选项，可用选项: {', '.join(valid_options.keys())}")
            
        # 应用设置
        setting_name = valid_options[option]
        setting_value = value_map[value.lower()]
        settings_manager.set_optimization_setting(group_id, setting_name, setting_value)
        
        # 特殊处理：开启高级语义匹配时训练模型
        if setting_name == "advanced_semantic" and setting_value:
            asyncio.create_task(learning_optimizer.train_advanced_model())
            await optimization_setting.finish(f"已启用高级语义匹配，模型训练中...")
            
        await optimization_setting.finish(f"已更新设置: {option} = {'开启' if setting_value else '关闭'}")
        
    except Exception as e:
        await optimization_setting.finish(f"设置失败: {str(e)}")

# ========== 性能优化命令 ==========
optimize_performance = on_command("优化性能", permission=SUPERUSER)
@optimize_performance.handle()
async def handle_optimize_performance(bot: Bot, event: MessageEvent):
    """执行性能优化操作"""
    try:
        # 预加载向量到内存
        await learning_optimizer.preload_message_vectors()
        # 训练高级模型
        await learning_optimizer.train_advanced_model()
        await optimize_performance.finish("✅ 性能优化完成：\n- 已预加载对话向量\n- 已训练高级语义模型")
    except Exception as e:
        await optimize_performance.finish(f"性能优化失败: {str(e)}")

# ========== 插件初始化 ==========
async def init_learning_assistant():
    """初始化学习助手插件"""
    logger.info("启动学习优化助手插件...")
    try:
        # 预加载资源
        await learning_optimizer.preload_message_vectors()
        
        # 根据设置决定是否训练高级模型
        if settings_manager.get_optimization_setting(0, "advanced_semantic", True):
            asyncio.create_task(learning_optimizer.train_advanced_model())
            
        logger.info("学习优化助手准备就绪")
    except Exception as e:
        logger.error(f"初始化失败: {e}")

# 注册启动钩子
driver = get_driver()
driver.on_startup(init_learning_assistant)