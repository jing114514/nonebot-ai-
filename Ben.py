import nonebot
from nonebot.adapters.onebot.v11 import (
    Bot, 
    MessageEvent, 
    GroupMessageEvent,
    PokeNotifyEvent, 
    Message,
    MessageSegment,
    GROUP_ADMIN,
    GROUP_OWNER
)
from nonebot.plugin import Plugin, on_message, on_notice, on_command
from nonebot.rule import to_me
from nonebot.permission import SUPERUSER
from nonebot.permission import Permission
from nonebot.params import CommandArg
import sqlite3
import os
import random
import time
import re
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set, Deque
import aiofiles
import httpx
from PIL import Image
import jieba
import jieba.analyse
from collections import deque
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import configparser
import logging
from nonebot.exception import FinishedException
from nonebot.adapters.onebot.v11.exception import ActionFailed

# ========== 配置管理 ==========
class Config:
    def __init__(self):
        # 创建基础路径
        self.data_path = Path.cwd() / "data" / "learning_chat_plus"
        self.emoji_path = self.data_path / "emojis"
        self.db_path = self.data_path / "chat_data.db"
        
        # 确保目录存在
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.emoji_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志 - 必须先初始化日志
        self.log_path = self.data_path / "learning_chat.log"
        self._setup_logging()
        
        # 加载配置文件
        self.config_file = self.data_path / "config.ini"
        self._load_config()
        
        # 初始化数据库
        self._init_db()
        
        # 加载停用词
        self.stop_words = self.load_stop_words()
        
        # 加载违规词
        self.blocked_words = self.load_blocked_words()
    
    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger("learning_chat")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("日志系统初始化完成")
    
    def _load_config(self):
        """加载或创建配置文件"""
        config = configparser.ConfigParser()
        
        # 默认配置 - 确保所有配置项都有默认值
        defaults = {
            'DEFAULT': {
                'poke_cooldown': '10',            # 戳一戳冷却时间(秒)
                'emoji_probability': '0.2',        # 表情包触发概率(0-1)
                'max_message_length': '100',       # 学习的最大消息长度
                'poke_back_probability': '0.3',    # 戳回概率(0-1)
                'poke_back_delay_min': '0.5',      # 戳回最小延迟(秒)
                'poke_back_delay_max': '2',        # 戳回最大延迟(秒)
                'max_keywords': '5',               # 最大关键词数量
                'at_reply_probability': '1',       # 艾特回复概率(0-1)
                'sequential_reply_probability': '1',  # 连续回复概率(0-1)
                'sequential_reply_delay_min': '0.5',    # 连续回复最小延迟(秒)
                'sequential_reply_delay_max': '1.5',    # 连续回复最大延迟(秒)
                'max_sequential_reply': '3',       # 最大连续回复条数
                'context_window': '3',             # 上下文记忆条数
                'min_similarity': '0.5',           # 最小相似度阈值(0-1)
                'keyword_weight': '0.6',           # 关键词权重(0-1)
                'context_weight': '0.6',           # 上下文权重(0-1)
                'response_diversity': '0.3',       # 回复多样性因子
                'min_keyword_length': '2',         # 最小关键词长度
                'enable_semantic_analysis': '1',    # 启用语义分析
                'min_message_length': '2',          # 最小消息长度
                'max_sql_variables': '500',         # SQL变量最大数量（修复关键点：提高到500）
                'at_reply_at_sender': '1',         # 是否在艾特回复时艾特发送者
                'at_reply_format': 'cq',           # 艾特格式: cq(原生艾特) 或 text(文本显示)
                'context_aware_reply': '1',        # 启用上下文感知回复
                'topic_coherence_weight': '0.7',   # 话题连贯性权重
                'response_length_factor': '0.8',   # 回复长度匹配因子
                'question_detection_enabled': '1', # 启用问题检测
                'emotional_response_enabled': '1', # 启用情感回应
                'conversation_depth': '2',         # 对话深度控制
                'blocked_words_enabled': '1',      # 启用违规词功能
                'message_quality_threshold': '0.6', # 消息质量阈值(0-1)
                'logical_thinking_enabled': '1',   # 启用逻辑思维增强
                'context_preservation_factor': '0.8', # 上下文保持因子
                'response_coherence_threshold': '0.6', # 回复连贯性阈值
                'conversation_memory_size': '10',   # 对话记忆大小
                'topic_transition_smoothness': '0.7', # 话题转换平滑度
                'response_logic_level': '2',        # 回复逻辑等级 (1-3)
                'default_group_speak_enabled': '1', # 默认群发言启用
                'default_group_learn_enabled': '1',  # 默认群学习启用
                'default_at_reply_enabled': '1',    # 默认艾特回复启用
                'default_poke_enabled': '1',        # 默认戳一戳回复启用
                'default_emoji_enabled': '1'        # 默认表情包功能启用
            }
        }
        
        # 如果配置文件不存在，则创建
        if not self.config_file.exists():
            config.read_dict(defaults)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                config.write(f)
            self.logger.info(f"已创建默认配置文件: {self.config_file}")
        
        # 读取配置
        config.read(self.config_file, encoding='utf-8')
        
        # 确保所有配置项都存在
        for section, options in defaults.items():
            for key, default_value in options.items():
                if not config.has_option(section, key):
                    config.set(section, key, default_value)
                    self.logger.warning(f"添加缺失配置项: {key} = {default_value}")
        
        # 保存更新后的配置
        with open(self.config_file, 'w', encoding='utf-8') as f:
            config.write(f)
        
        # 获取配置值
        self.poke_cooldown = float(config.get('DEFAULT', 'poke_cooldown'))
        self.emoji_probability = float(config.get('DEFAULT', 'emoji_probability'))
        self.max_message_length = int(config.get('DEFAULT', 'max_message_length'))
        self.min_message_length = int(config.get('DEFAULT', 'min_message_length'))
        self.emoji_formats = ['.jpg', '.jpeg', '.png', '.gif']
        self.poke_back_probability = float(config.get('DEFAULT', 'poke_back_probability'))
        self.poke_back_delay_min = float(config.get('DEFAULT', 'poke_back_delay_min'))
        self.poke_back_delay_max = float(config.get('DEFAULT', 'poke_back_delay_max'))
        self.poke_back_delay = (self.poke_back_delay_min, self.poke_back_delay_max)
        self.max_keywords = int(config.get('DEFAULT', 'max_keywords'))
        self.at_reply_probability = float(config.get('DEFAULT', 'at_reply_probability'))
        self.sequential_reply_probability = float(config.get('DEFAULT', 'sequential_reply_probability'))
        self.sequential_reply_delay_min = float(config.get('DEFAULT', 'sequential_reply_delay_min'))
        self.sequential_reply_delay_max = float(config.get('DEFAULT', 'sequential_reply_delay_max'))
        self.sequential_reply_delay = (self.sequential_reply_delay_min, self.sequential_reply_delay_max)
        self.max_sequential_reply = int(config.get('DEFAULT', 'max_sequential_reply'))
        self.context_window = int(config.get('DEFAULT', 'context_window'))
        self.min_similarity = float(config.get('DEFAULT', 'min_similarity'))
        self.keyword_weight = float(config.get('DEFAULT', 'keyword_weight'))
        self.context_weight = float(config.get('DEFAULT', 'context_weight'))
        self.response_diversity = float(config.get('DEFAULT', 'response_diversity'))
        self.min_keyword_length = int(config.get('DEFAULT', 'min_keyword_length'))
        self.enable_semantic_analysis = int(config.get('DEFAULT', 'enable_semantic_analysis'))
        self.max_sql_variables = int(config.get('DEFAULT', 'max_sql_variables'))
        self.at_reply_at_sender = int(config.get('DEFAULT', 'at_reply_at_sender'))
        self.at_reply_format = config.get('DEFAULT', 'at_reply_format')
        self.context_aware_reply = int(config.get('DEFAULT', 'context_aware_reply'))
        self.topic_coherence_weight = float(config.get('DEFAULT', 'topic_coherence_weight'))
        self.response_length_factor = float(config.get('DEFAULT', 'response_length_factor'))
        self.question_detection_enabled = int(config.get('DEFAULT', 'question_detection_enabled'))
        self.emotional_response_enabled = int(config.get('DEFAULT', 'emotional_response_enabled'))
        self.conversation_depth = int(config.get('DEFAULT', 'conversation_depth'))
        self.blocked_words_enabled = int(config.get('DEFAULT', 'blocked_words_enabled'))
        self.message_quality_threshold = float(config.get('DEFAULT', 'message_quality_threshold'))
        self.logical_thinking_enabled = int(config.get('DEFAULT', 'logical_thinking_enabled'))
        self.context_preservation_factor = float(config.get('DEFAULT', 'context_preservation_factor'))
        self.response_coherence_threshold = float(config.get('DEFAULT', 'response_coherence_threshold'))
        self.conversation_memory_size = int(config.get('DEFAULT', 'conversation_memory_size'))
        self.topic_transition_smoothness = float(config.get('DEFAULT', 'topic_transition_smoothness'))
        self.response_logic_level = int(config.get('DEFAULT', 'response_logic_level'))
        self.default_group_speak_enabled = int(config.get('DEFAULT', 'default_group_speak_enabled'))
        self.default_group_learn_enabled = int(config.get('DEFAULT', 'default_group_learn_enabled'))
        self.default_at_reply_enabled = int(config.get('DEFAULT', 'default_at_reply_enabled'))
        self.default_poke_enabled = int(config.get('DEFAULT', 'default_poke_enabled'))
        self.default_emoji_enabled = int(config.get('DEFAULT', 'default_emoji_enabled'))
        
        self.logger.info("配置加载完成")
    
    def _init_db(self):
        """初始化数据库结构并修复缺失列 - 全局数据管理"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 1. 全局聊天日志表 (添加group_id字段)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL DEFAULT 0,
                    user_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    keywords TEXT,
                    quality_score REAL DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # 检查并添加缺失列
                cursor.execute("PRAGMA table_info(chat_logs)")
                columns = [row[1] for row in cursor.fetchall()]
                
                # 添加group_id列（如果不存在）
                if 'group_id' not in columns:
                    cursor.execute("ALTER TABLE chat_logs ADD COLUMN group_id INTEGER NOT NULL DEFAULT 0")
                    self.logger.info("已添加缺失列: chat_logs.group_id")
                
                # 添加keywords列（如果不存在）
                if 'keywords' not in columns:
                    cursor.execute("ALTER TABLE chat_logs ADD COLUMN keywords TEXT")
                    self.logger.info("已添加缺失列: chat_logs.keywords")
                
                # 添加quality_score列（如果不存在）
                if 'quality_score' not in columns:
                    cursor.execute("ALTER TABLE chat_logs ADD COLUMN quality_score REAL DEFAULT 1.0")
                    self.logger.info("已添加缺失列: chat_logs.quality_score")
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON chat_logs(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_group ON chat_logs(group_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON chat_logs(quality_score)")
                
                # 2. 表情包表 (保留group_id用于特殊表情)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS emojis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER DEFAULT 0,
                    file_path TEXT UNIQUE NOT NULL,
                    trigger_words TEXT,
                    added_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trigger ON emojis(trigger_words)")
                
                # 3. 群组设置表 (功能开关仍按群管理)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS group_settings (
                    group_id INTEGER PRIMARY KEY,
                    learning_enabled BOOLEAN DEFAULT 1,
                    poke_enabled BOOLEAN DEFAULT 1,
                    emoji_enabled BOOLEAN DEFAULT 1,
                    at_reply_enabled BOOLEAN DEFAULT 1,
                    ai_mode BOOLEAN DEFAULT 1,
                    speak_enabled BOOLEAN DEFAULT 1,  -- 新增：发言功能
                    learn_enabled BOOLEAN DEFAULT 1   -- 新增：学习功能
                )
                """)
                
                # 检查并添加缺失列
                cursor.execute("PRAGMA table_info(group_settings)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'ai_mode' not in columns:
                    cursor.execute("ALTER TABLE group_settings ADD COLUMN ai_mode BOOLEAN DEFAULT 1")
                    self.logger.info("已添加缺失列: group_settings.ai_mode")
                if 'speak_enabled' not in columns:
                    cursor.execute("ALTER TABLE group_settings ADD COLUMN speak_enabled BOOLEAN DEFAULT 1")
                    self.logger.info("已添加缺失列: group_settings.speak_enabled")
                if 'learn_enabled' not in columns:
                    cursor.execute("ALTER TABLE group_settings ADD COLUMN learn_enabled BOOLEAN DEFAULT 1")
                    self.logger.info("已添加缺失列: group_settings.learn_enabled")
                
                # 4. 关键词关联表 (全局关键词) - 修复主键约束
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS keyword_associations (
                    keyword TEXT NOT NULL,
                    associated_keyword TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (keyword, associated_keyword)
                )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_keyword ON keyword_associations(keyword)")
                
                # 5. 对话上下文表 (新增)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_contexts (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    context_text TEXT NOT NULL,
                    topic TEXT,
                    sentiment REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_id, user_id)
                )
                """)
                
                # 6. 违规词表 (新增)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocked_words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL,
                    added_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_word ON blocked_words(word)")
                
                # 7. 对话逻辑表 (新增)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_logic (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_pattern TEXT NOT NULL,
                    response_pattern TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # 8. 对话记忆表 (新增)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    group_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_id, user_id, memory_key)
                )
                """)
                
                conn.commit()
                self.logger.info("数据库初始化完成")
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    def load_stop_words(self) -> Set[str]:
        """加载停用词表"""
        stop_words_path = self.data_path / "stop_words.txt"
        try:
            if not stop_words_path.exists():
                # 创建默认停用词表
                default_stop_words = [
                    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
                    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"
                ]
                with open(stop_words_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(default_stop_words))
                self.logger.info(f"已创建默认停用词表: {stop_words_path}")
            
            with open(stop_words_path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            self.logger.error(f"加载停用词失败: {e}")
            return set()
    
    def load_blocked_words(self) -> Set[str]:
        """加载违规词表"""
        blocked_words_path = self.data_path / "blocked_words.txt"
        try:
            if not blocked_words_path.exists():
                # 创建默认违规词表
                default_blocked_words = [
                    # 默认屏蔽一些无意义内容
                    "签到", "打卡", "分享", "点击", "查看", "领取", "活动",
                    "http://", "https://", "www.", ".com", ".cn", ".net",
                    "[图片]", "[视频]", "[语音]", "[文件]", "[动画表情]"
                ]
                with open(blocked_words_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(default_blocked_words))
                self.logger.info(f"已创建默认违规词表: {blocked_words_path}")
            
            with open(blocked_words_path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            self.logger.error(f"加载违规词失败: {e}")
            return set()
    
    def save_blocked_words(self):
        """保存违规词到文件"""
        blocked_words_path = self.data_path / "blocked_words.txt"
        try:
            with open(blocked_words_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.blocked_words))
            self.logger.info(f"已保存违规词表: {blocked_words_path}")
        except Exception as e:
            self.logger.error(f"保存违规词失败: {e}")

config = Config()
logger = config.logger

# 初始化jieba分词
jieba.initialize()
# ============================

# ========== 增强AI核心功能 ==========
class EnhancedAI:
    def __init__(self):
        # 上下文记忆: {group_id: {user_id: deque}}
        self.context_memory = {}
        # TF-IDF向量化器 (全局)
        self.vectorizer = None
        # 关键词关联缓存
        self.keyword_cache = {}
        # 语义模型训练状态
        self.model_trained = False
        # 消息缓存 (用于训练)
        self.message_cache = deque(maxlen=5000)
        # 对话主题缓存
        self.topic_cache = {}
        # 情感分析缓存
        self.sentiment_cache = {}
        # 对话逻辑缓存
        self.conversation_logic = {}
        # 对话状态跟踪
        self.conversation_states = {}
        # 对话记忆
        self.conversation_memory = {}
        
        # 问题模式
        self.question_patterns = [
            r"(.+?)[吗嘛呢吧啊]?[？?]$",
            r"为什么(.+)",
            r"怎么(.+)",
            r"如何(.+)",
            r"是不是(.+)",
            r"有没有(.+)",
            r"可否(.+)",
            r"能否(.+)",
            r"要不要(.+)",
            r"该不该(.+)",
            r"是不是(.+)",
            r"会不会(.+)"
        ]
        
        # 情感词库
        self.emotional_words = {
            "开心": ["高兴", "快乐", "开心", "喜悦", "愉快", "兴奋"],
            "难过": ["伤心", "难过", "悲伤", "沮丧", "失望", "郁闷"],
            "生气": ["生气", "愤怒", "恼火", "气愤", "不爽", "烦躁"],
            "惊讶": ["惊讶", "惊奇", "震惊", "意外", "吃惊", "没想到"],
            "喜欢": ["喜欢", "爱", "欣赏", "迷恋", "钟意", "宠爱"],
            "讨厌": ["讨厌", "厌恶", "反感", "憎恶", "嫌弃", "不喜欢"]
        }
        
        # 消息质量评估模式
        self.low_quality_patterns = [
            r"^[^\u4e00-\u9fa5a-zA-Z0-9]{3,}$",  # 纯符号
            r"^.{1,2}$",  # 过短消息
            r"(.)\1{4,}",  # 重复字符过多
            r"[0-9]{6,}",  # 长数字串
            r"[a-zA-Z]{10,}",  # 长英文字母串
        ]
        
        # 对话逻辑模式
        self._init_conversation_logic()
        
        # 逻辑连接词
        self.logical_connectors = {
            "因果": ["因为", "所以", "因此", "于是", "导致", "结果"],
            "转折": ["但是", "可是", "然而", "不过", "却", "虽然"],
            "递进": ["而且", "并且", "甚至", "更", "还", "另外"],
            "条件": ["如果", "要是", "假如", "只要", "除非", "无论"],
            "选择": ["或者", "还是", "要么", "不是", "就是"],
            "总结": ["总之", "总而言之", "总的来说", "综上所述"]
        }
    
    def _init_conversation_logic(self):
        """初始化对话逻辑模式"""
        # 常见对话模式
        self.conversation_logic = {
            # 问候模式
            r"(早上好|早安|早啊)": ["早上好呀~", "早安！今天也是充满希望的一天呢", "早啊，睡得好吗？"],
            r"(晚上好|晚安|晚啊)": ["晚上好~", "晚安，做个好梦", "晚啊，今天过得怎么样？"],
            
            # 感谢模式
            r"(谢谢|感谢|多谢)": ["不客气~", "举手之劳", "能帮到你就好"],
            
            # 道歉模式
            r"(对不起|抱歉|不好意思)": ["没关系~", "不用在意", "我理解"],
            
            # 疑问模式
            r"(是吗|真的吗|真的假的)": ["当然是真的啦", "我怎么会骗你呢", "信不信由你咯"],
            
            # 肯定模式
            r"(好的|好吧|可以)": ["太好了", "就这么说定了", "好的呢"],
            
            # 否定模式
            r"(不要|不行|不可以)": ["为什么呀", "那好吧", "再考虑考虑嘛"],
        }
    
    async def is_group_ai_enabled(self, group_id: int) -> bool:
        """检查群组AI模式是否启用"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT ai_mode FROM group_settings WHERE group_id=?",
                    (group_id,)
                )
                result = cursor.fetchone()
                # 如果查询失败或没有设置，默认开启
                return result[0] if result else True
        except Exception as e:
            logger.error(f"检查AI模式失败: {e}")
            return True
    
    def update_context(self, group_id: int, user_id: int, message: str):
        """更新上下文记忆 - 增强逻辑性"""
        key = (group_id, user_id)
        if key not in self.context_memory:
            self.context_memory[key] = deque(maxlen=config.context_window)
        
        # 添加新消息到上下文
        self.context_memory[key].append(message)
        
        # 更新对话主题
        self._update_conversation_topic(group_id, user_id, message)
        
        # 更新情感分析
        if config.emotional_response_enabled:
            self._update_sentiment_analysis(group_id, user_id, message)
        
        # 更新对话状态
        self._update_conversation_state(group_id, user_id, message)
        
        # 更新对话记忆
        self._update_conversation_memory(group_id, user_id, message)
    
    def _update_conversation_topic(self, group_id: int, user_id: int, message: str):
        """更新对话主题分析"""
        key = (group_id, user_id)
        keywords = self.extract_keywords(message, top_n=3)
        
        if keywords:
            if key not in self.topic_cache:
                self.topic_cache[key] = deque(maxlen=3)
            
            # 添加关键词到主题缓存
            self.topic_cache[key].extend(keywords)
    
    def _update_sentiment_analysis(self, group_id: int, user_id: int, message: str):
        """更新情感分析"""
        key = (group_id, user_id)
        
        # 简单情感分析 - 检测情感词
        sentiment_score = 0
        for emotion, words in self.emotional_words.items():
            for word in words:
                if word in message:
                    if emotion in ["开心", "喜欢"]:
                        sentiment_score += 1
                    elif emotion in ["难过", "生气", "讨厌"]:
                        sentiment_score -= 1
        
        # 平滑情感变化
        if key in self.sentiment_cache:
            old_score = self.sentiment_cache[key]
            self.sentiment_cache[key] = (old_score * 0.7) + (sentiment_score * 0.3)
        else:
            self.sentiment_cache[key] = sentiment_score
    
    def _update_conversation_state(self, group_id: int, user_id: int, message: str):
        """更新对话状态"""
        key = (group_id, user_id)
        
        # 初始化对话状态
        if key not in self.conversation_states:
            self.conversation_states[key] = {
                "topic": "",
                "question_count": 0,
                "response_count": 0,
                "last_interaction": time.time(),
                "conversation_depth": 0
            }
        
        # 更新对话状态
        state = self.conversation_states[key]
        state["last_interaction"] = time.time()
        
        # 检测是否是问题
        if self.detect_question(message):
            state["question_count"] += 1
            state["conversation_depth"] = min(state["conversation_depth"] + 1, config.conversation_depth)
        else:
            # 逐渐减少对话深度
            state["conversation_depth"] = max(state["conversation_depth"] - 0.5, 0)
        
        # 更新话题
        keywords = self.extract_keywords(message, top_n=2)
        if keywords:
            state["topic"] = " ".join(keywords)
    
    def _update_conversation_memory(self, group_id: int, user_id: int, message: str):
        """更新对话记忆"""
        key = (group_id, user_id)
        
        if key not in self.conversation_memory:
            self.conversation_memory[key] = deque(maxlen=config.conversation_memory_size)
        
        # 提取重要信息存入记忆
        important_info = self._extract_important_info(message)
        if important_info:
            self.conversation_memory[key].append(important_info)
    
    def _extract_important_info(self, message: str) -> Optional[str]:
        """从消息中提取重要信息"""
        # 检测是否包含重要信息（如数字、时间、地点等）
        important_patterns = [
            r"\d+",  # 数字
            r"今天|明天|昨天|现在|以后|之前",  # 时间
            r"这里|那里|地方|位置",  # 地点
            r"喜欢|讨厌|爱|恨|想要|需要",  # 情感偏好
        ]
        
        for pattern in important_patterns:
            if re.search(pattern, message):
                return message
        
        return None
    
    def get_context(self, group_id: int, user_id: int) -> List[str]:
        """获取上下文记忆"""
        key = (group_id, user_id)
        return list(self.context_memory.get(key, []))
    
    def get_topic(self, group_id: int, user_id: int) -> List[str]:
        """获取对话主题"""
        key = (group_id, user_id)
        if key in self.topic_cache:
            # 返回去重后的主题词
            return list(set(self.topic_cache[key]))
        return []
    
    def get_sentiment(self, group_id: int, user_id: int) -> int:
        """获取情感分数"""
        key = (group_id, user_id)
        return self.sentiment_cache.get(key, 0)
    
    def get_conversation_state(self, group_id: int, user_id: int) -> Dict:
        """获取对话状态"""
        key = (group_id, user_id)
        return self.conversation_states.get(key, {})
    
    def get_conversation_memory(self, group_id: int, user_id: int) -> List[str]:
        """获取对话记忆"""
        key = (group_id, user_id)
        return list(self.conversation_memory.get(key, []))
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词 - 增强版"""
        try:
            # 过滤特殊符号
            cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]+', ' ', text)
            
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(
                cleaned_text, 
                topK=top_n, 
                withWeight=False,
                allowPOS=('n', 'vn', 'v', 'a')
            )
            
            # 过滤停用词和短词
            return [
                kw for kw in keywords 
                if (kw not in config.stop_words and 
                    len(kw) >= config.min_keyword_length)
            ]
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []
    
    def detect_question(self, text: str) -> bool:
        """检测是否为问题"""
        if not config.question_detection_enabled:
            return False
            
        # 检查是否包含疑问词
        question_words = ["吗", "么", "呢", "吧", "?", "？", "为什么", "怎么", "如何", "是不是", "有没有", "可否", "能否", "要不要", "该不该"]
        if any(word in text for word in question_words):
            return True
            
        # 检查疑问模式
        for pattern in self.question_patterns:
            if re.match(pattern, text):
                return True
                
        return False
    
    def generate_emotional_response(self, sentiment: int, context: List[str] = None) -> Optional[str]:
        """生成情感回应"""
        if sentiment > 1:
            # 积极情感回应
            positive_responses = [
                "听起来你很开心呢！",
                "真好！看到你高兴我也很开心~",
                "太棒了！继续保持好心情！"
            ]
            return random.choice(positive_responses)
        elif sentiment < -1:
            # 消极情感回应
            negative_responses = [
                "怎么了？看起来你有点不开心...",
                "别难过，一切都会好起来的~",
                "有什么烦恼可以和我聊聊哦"
            ]
            return random.choice(negative_responses)
        return None
    
    def assess_message_quality(self, text: str) -> float:
        """评估消息质量 (0-1)"""
        # 基础质量分数
        quality = 1.0
        
        # 检查低质量模式
        for pattern in self.low_quality_patterns:
            if re.search(pattern, text):
                quality *= 0.5  # 降低质量分数
        
        # 检查违规词
        if config.blocked_words_enabled and any(word in text for word in config.blocked_words):
            quality *= 0.3  # 大幅降低质量分数
        
        # 检查消息长度
        length = len(text)
        if length < 3:
            quality *= 0.2  # 过短消息质量低
        elif length > config.max_message_length:
            quality *= 0.7  # 过长消息质量稍低
        
        # 检查重复字符
        if re.search(r"(.)\1{3,}", text):
            quality *= 0.4  # 重复字符过多
        
        # 检查中文比例
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        chinese_ratio = len(chinese_chars) / max(1, len(text))
        if chinese_ratio < 0.3 and len(text) > 5:
            quality *= 0.6  # 中文比例过低
        
        return max(0.1, min(1.0, quality))  # 确保在0.1-1.0范围内
    
    async def update_keyword_associations(self, message: str, message_id: int, group_id: int, quality_score: float = 1.0):
        """更新全局关键词关联 - 修复ON CONFLICT问题"""
        try:
            keywords = self.extract_keywords(message)
            if not keywords:
                return
            
            # 保存关键词到数据库
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                # 更新消息的关键词字段和质量分数
                cursor.execute(
                    "UPDATE chat_logs SET keywords=?, quality_score=? WHERE id=?",
                    (json.dumps(keywords), quality_score, message_id)
                )
                
                # 更新关键词关联 - 使用UPSERT语法修复冲突
                for i in range(len(keywords)):
                    for j in range(i + 1, len(keywords)):
                        kw1 = keywords[i]
                        kw2 = keywords[j]
                        
                        # 确保kw1和kw2不同
                        if kw1 == kw2:
                            continue
                        
                        # 修复ON CONFLICT问题：使用REPLACE INTO替代INSERT OR REPLACE
                        cursor.execute(
                            """REPLACE INTO keyword_associations (keyword, associated_keyword, weight) 
                            VALUES (?, ?, COALESCE((SELECT weight FROM keyword_associations 
                                                    WHERE keyword=? AND associated_keyword=?), 0) + 0.1 * ?)""",
                            (kw1, kw2, kw1, kw2, quality_score)
                        )
                
                conn.commit()
            
            # 更新缓存
            self.update_keyword_cache()
        except Exception as e:
            logger.error(f"更新关键词关联失败: {e}")
    
    def update_keyword_cache(self):
        """更新关键词缓存"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT keyword, associated_keyword, weight FROM keyword_associations"
                )
                results = cursor.fetchall()
                
                self.keyword_cache = {}
                for kw, assoc_kw, weight in results:
                    if kw not in self.keyword_cache:
                        self.keyword_cache[kw] = {}
                    self.keyword_cache[kw][assoc_kw] = weight
        except Exception as e:
            logger.error(f"更新关键词缓存失败: {e}")
            self.keyword_cache = {}
    
    def get_related_keywords(self, keyword: str, top_n: int = 3) -> List[str]:
        """获取相关关键词"""
        if not self.keyword_cache:
            self.update_keyword_cache()
        
        if keyword not in self.keyword_cache:
            return []
        
        # 按权重排序
        related = sorted(
            self.keyword_cache[keyword].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [kw for kw, _ in related]
    
    async def train_vectorizer(self):
        """训练全局TF-IDF向量化器"""
        try:
            # 如果消息缓存足够多，优先使用缓存
            if len(self.message_cache) > 1000:
                messages = list(self.message_cache)
            else:
            # 否则从数据库加载高质量消息
                with sqlite3.connect(config.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT message FROM chat_logs WHERE quality_score >= ?", 
                                  (config.message_quality_threshold,))
                    results = cursor.fetchall()
                    messages = [row[0] for row in results]
            
            if not messages:
                return
                
            # 创建TF-IDF向量化器
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(messages)
            self.model_trained = True
            logger.info(f"全局TF-IDF模型训练完成，样本数: {len(messages)}")
        except Exception as e:
            logger.error(f"训练向量化器失败: {e}")
    
    async def find_most_relevant(self, query: str, context: List[str] = None, topic: List[str] = None) -> Optional[str]:
        """找到最相关的回复 - 增强版"""
        # 确保向量化器已训练
        if not self.vectorizer or not self.model_trained:
            await self.train_vectorizer()
        
        if not self.vectorizer:
            return None
        
        try:
            # 组合查询和上下文
            full_query = " ".join(context) + " " + query if context else query
            
            # 添加主题关键词
            if topic and config.context_aware_reply:
                full_query += " " + " ".join(topic)
            
            # 向量化查询
            query_vec = self.vectorizer.transform([full_query])
            
            # 获取高质量消息
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, message FROM chat_logs WHERE quality_score >= ?", 
                              (config.message_quality_threshold,))
                results = cursor.fetchall()
                
                if not results:
                    return None
                
                # 向量化所有消息
                messages = [row[1] for row in results]
                message_vectors = self.vectorizer.transform(messages)
                
                # 计算相似度
                similarities = cosine_similarity(query_vec, message_vectors)
                
                # 引入多样性：不只是选择最相似的，而是从最相似的前5个中随机选择
                top_indices = np.argsort(similarities[0])[-5:][::-1]
                
                # 根据配置的多样性因子决定是否随机选择
                if random.random() < config.response_diversity and len(top_indices) > 1:
                    # 从最相似的前几个中随机选择
                    best_index = random.choice(top_indices[:3])
                else:
                    # 选择最相似的
                    best_index = top_indices[0]
                
                max_similarity = similarities[0, best_index]
                
                # 检查相似度阈值
                if max_similarity < config.min_similarity:
                    return None
                
                return results[best_index][1]
        except Exception as e:
            logger.error(f"查找相关回复失败: {e}")
            return None
    
    def _check_conversation_logic(self, message: str) -> Optional[str]:
        """检查对话逻辑模式"""
        if not config.logical_thinking_enabled:
            return None
            
        for pattern, responses in self.conversation_logic.items():
            if re.search(pattern, message):
                return random.choice(responses)
        return None
    
    def _enhance_response_coherence(self, response: str, context: List[str], topic: List[str]) -> str:
        """增强回复的连贯性"""
        if not context or not response:
            return response
        
        # 检查回复是否与上下文相关
        context_text = " ".join(context)
        response_keywords = set(self.extract_keywords(response))
        context_keywords = set(self.extract_keywords(context_text))
        
        # 计算关键词重叠度
        overlap = len(response_keywords & context_keywords) / max(1, len(response_keywords))
        
        # 如果重叠度太低，尝试找到更相关的回复
        if overlap < config.response_coherence_threshold:
            # 尝试从关键词关联中寻找更相关的回复
            expanded_keywords = []
            for kw in context_keywords:
                expanded_keywords.extend(self.get_related_keywords(kw))
            
            # 如果有扩展关键词，尝试重新生成回复
            if expanded_keywords:
                # 这里可以调用其他方法来生成更相关的回复
                pass
        
        return response
    
    def _add_logical_connectors(self, response: str, context: List[str]) -> str:
        """为回复添加逻辑连接词"""
        if not context or random.random() > 0.3:  # 30%概率添加逻辑词
            return response
        
        # 根据上下文选择合适的逻辑词
        last_message = context[-1] if context else ""
        
        # 检测最后一条消息的逻辑关系
        connector_type = None
        for conn_type, words in self.logical_connectors.items():
            for word in words:
                if word in last_message:
                    connector_type = conn_type
                    break
            if connector_type:
                break
        
        # 根据逻辑关系选择合适的连接词
        if connector_type:
            connectors = {
                "因果": ["所以", "因此", "于是"],
                "转折": ["但是", "不过", "然而"],
                "递进": ["而且", "并且", "另外"],
                "条件": ["如果这样", "要是这样的话"],
                "选择": ["或者", "还是"],
                "总结": ["总之", "总的来说"]
            }
            
            if connector_type in connectors:
                connector = random.choice(connectors[connector_type])
                return f"{connector}，{response}"
        
        return response
    
    def _improve_response_flow(self, response: str, conversation_state: Dict) -> str:
        """改进回复的流畅度和自然度"""
        # 根据对话深度调整回复风格
        depth = conversation_state.get("conversation_depth", 0)
        
        if depth > 1:
            # 深度对话时，使用更详细的回复
            if len(response) < 10 and random.random() < 0.5:
                elaborations = [
                    "我觉得这个事情",
                    "关于这个",
                    "其实我想说",
                    "我个人认为"
                ]
                response = f"{random.choice(elaborations)}，{response}"
        
        # 添加语气词使回复更自然
        if random.random() < 0.2:  # 20%概率添加语气词
            modal_particles = ["呢", "呀", "啊", "啦", "喔"]
            if response[-1] not in modal_particles and len(response) < 15:
                response += random.choice(modal_particles)
        
        return response
    
    async def generate_response(self, group_id: int, user_id: int, message: str) -> str:
        """生成AI风格的回复 - 增强逻辑性版本"""
        # 0. 检查AI模式是否启用
        if not await self.is_group_ai_enabled(group_id):
            return await self.get_random_history()
        
        # 1. 获取上下文和主题
        context = self.get_context(group_id, user_id)
        topic = self.get_topic(group_id, user_id)
        sentiment = self.get_sentiment(group_id, user_id)
        conversation_state = self.get_conversation_state(group_id, user_id)
        conversation_memory = self.get_conversation_memory(group_id, user_id)
        
        # 2. 检查对话逻辑模式
        logic_response = self._check_conversation_logic(message)
        if logic_response:
            return logic_response
        
        # 3. 情感回应
        if config.emotional_response_enabled and random.random() < 0.3:
            emotional_response = self.generate_emotional_response(sentiment, context)
            if emotional_response:
                return emotional_response
        
        # 4. 问题检测与回应
        if self.detect_question(message):
            # 尝试找到相关问题的最佳回答
            question_reply = await self.find_question_reply(message, context, topic)
            if question_reply:
                # 增强回复连贯性
                enhanced_reply = self._enhance_response_coherence(question_reply, context, topic)
                # 添加逻辑连接词
                enhanced_reply = self._add_logical_connectors(enhanced_reply, context)
                # 改进回复流畅度
                enhanced_reply = self._improve_response_flow(enhanced_reply, conversation_state)
                return enhanced_reply
        
        # 5. 尝试找到最相关的回复 (使用全局数据)
        if config.enable_semantic_analysis:
            relevant_reply = await self.find_most_relevant(message, context, topic)
            if relevant_reply:
                # 检查回复长度是否匹配
                if self._is_length_appropriate(message, relevant_reply):
                    # 增强回复连贯性
                    enhanced_reply = self._enhance_response_coherence(relevant_reply, context, topic)
                    # 添加逻辑连接词
                    enhanced_reply = self._add_logical_connectors(enhanced_reply, context)
                    # 改进回复流畅度
                    enhanced_reply = self._improve_response_flow(enhanced_reply, conversation_state)
                    return enhanced_reply
        
        # 6. 基于关键词生成新回复
        keywords = self.extract_keywords(message)
        if keywords:
            # 尝试扩展关键词
            expanded_keywords = []
            for kw in keywords:
                expanded_keywords.append(kw)
                expanded_keywords.extend(self.get_related_keywords(kw))
            
            # 添加主题关键词
            if topic:
                expanded_keywords.extend(topic)
            
            # 去重
            expanded_keywords = list(set(expanded_keywords))
            
            # 尝试找到包含这些关键词的消息
            keyword_reply = await self.get_keyword_reply(expanded_keywords)
            if keyword_reply and self._is_length_appropriate(message, keyword_reply):
                # 增强回复连贯性
                enhanced_reply = self._enhance_response_coherence(keyword_reply, context, topic)
                # 添加逻辑连接词
                enhanced_reply = self._add_logical_connectors(enhanced_reply, context)
                # 改进回复流畅度
                enhanced_reply = self._improve_response_flow(enhanced_reply, conversation_state)
                return enhanced_reply
        
        # 7. 作为后备方案，返回随机历史消息
        fallback_reply = await self.get_random_history()
        if fallback_reply:
            # 增强回复连贯性
            enhanced_reply = self._enhance_response_coherence(fallback_reply, context, topic)
            # 添加逻辑连接词
            enhanced_reply = self._add_logical_connectors(enhanced_reply, context)
            # 改进回复流畅度
            enhanced_reply = self._improve_response_flow(enhanced_reply, conversation_state)
            return enhanced_reply
        
        # 8. 最终后备方案
        default_responses = [
            "嗯...我不太明白呢",
            "这个话题有点难理解",
            "能换个方式说说吗？",
            "我不太确定该怎么回答"
        ]
        return random.choice(default_responses)
    
    def _is_length_appropriate(self, query: str, response: str) -> bool:
        """检查回复长度是否合适"""
        query_len = len(query)
        response_len = len(response)
        
        # 如果查询很短，回复也不应该太长
        if query_len < 10 and response_len > 30:
            return False
            
        # 如果查询很长，回复也不应该太短
        if query_len > 30 and response_len < 10:
            return False
            
        # 长度差异因子
        length_ratio = response_len / max(query_len, 1)
        return 0.5 <= length_ratio <= 2.0
    
    async def find_question_reply(self, question: str, context: List[str] = None, topic: List[str] = None) -> Optional[str]:
        """专门针对问题寻找回复"""
        try:
            # 提取问题关键词
            question_keywords = self.extract_keywords(question, top_n=3)
            
            # 组合查询条件
            query_terms = question_keywords
            if topic:
                query_terms.extend(topic)
            
            # 去重
            query_terms = list(set(query_terms))
            
            # 尝试找到最佳回答
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                for term in query_terms:
                    conditions.append("message LIKE ?")
                    params.append(f"%{term}%")
                
                if conditions:
                    query = " OR ".join(conditions)
                    cursor.execute(
                        f"SELECT message FROM chat_logs WHERE ({query}) AND quality_score >= ? ORDER BY RANDOM() LIMIT 5",
                        params + [config.message_quality_threshold]
                    )
                    results = cursor.fetchall()
                    
                    if results:
                        # 选择最合适的回答
                        best_reply = self._select_best_reply_for_question(question, [r[0] for r in results])
                        return best_reply
                
                return None
        except Exception as e:
            logger.error(f"寻找问题回复失败: {e}")
            return None
    
    def _select_best_reply_for_question(self, question: str, candidates: List[str]) -> str:
        """从候选回复中选择最适合问题的回答"""
        # 简单策略：选择包含疑问词或长度适中的回复
        question_indicators = ["是", "不是", "可以", "不能", "会", "不会", "有", "没有", "能", "不能"]
        
        for candidate in candidates:
            # 优先选择包含疑问词回答的候选
            if any(indicator in candidate for indicator in question_indicators):
                return candidate
        
        # 其次选择长度适中的候选
        question_len = len(question)
        appropriate_candidates = [
            c for c in candidates 
            if 0.7 * question_len <= len(c) <= 1.5 * question_len
        ]
        
        if appropriate_candidates:
            return random.choice(appropriate_candidates)
        
        # 最后随机选择
        return random.choice(candidates)
    
    async def get_keyword_reply(self, keywords: List[str]) -> Optional[str]:
        """基于关键词获取回复 - 全局搜索 (修复关键点：分批处理SQL查询)"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                # 检查keywords列是否存在
                cursor.execute("PRAGMA table_info(chat_logs)")
                columns = [row[1] for row in cursor.fetchall()]
                use_keywords = 'keywords' in columns
                
                # 限制关键词数量，避免SQL变量过多
                max_vars = config.max_sql_variables // 2
                if len(keywords) > max_vars:
                    logger.warning(f"关键词数量超过限制 ({len(keywords)} > {max_vars})，已截断")
                    keywords = keywords[:max_vars]
                
                if use_keywords and keywords:
                    # 分批处理关键词
                    batch_size = config.max_sql_variables // 2
                    for i in range(0, len(keywords), batch_size):
                        batch_keywords = keywords[i:i+batch_size]
                        
                        # 构建查询条件
                        conditions = []
                        params = []
                        for kw in batch_keywords:
                            conditions.append("keywords LIKE ?")
                            params.append(f"%{kw}%")
                        
                        # 执行查询
                        query = " OR ".join(conditions)
                        cursor.execute(
                            f"SELECT message FROM chat_logs WHERE ({query}) AND quality_score >= ? ORDER BY RANDOM() LIMIT 1",
                            params + [config.message_quality_threshold]
                        )
                        result = cursor.fetchone()
                        if result:
                            return result[0]
                
                # 如果keywords列不存在或查询失败，使用消息内容匹配
                if keywords:
                    # 分批处理关键词
                    batch_size = config.max_sql_variables // 2
                    for i in range(0, len(keywords), batch_size):
                        batch_keywords = keywords[i:i+batch_size]
                        
                        conditions = []
                        params = []
                        for kw in batch_keywords:
                            conditions.append("message LIKE ?")
                            params.append(f"%{kw}%")
                        
                        query = " OR ".join(conditions)
                        cursor.execute(
                            f"SELECT message FROM chat_logs WHERE ({query}) AND quality_score >= ? ORDER BY RANDOM() LIMIT 1",
                            params + [config.message_quality_threshold]
                        )
                        result = cursor.fetchone()
                        if result:
                            return result[0]
                
                # 最后尝试随机获取高质量消息
                cursor.execute(
                    "SELECT message FROM chat_logs WHERE quality_score >= ? ORDER BY RANDOM() LIMIT 1",
                    (config.message_quality_threshold,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"获取关键词回复失败: {e}")
            return None
    
    async def get_random_history(self) -> Optional[str]:
        """从数据库随机获取历史消息 - 全局"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT message FROM chat_logs WHERE quality_score >= ? ORDER BY RANDOM() LIMIT 1",
                    (config.message_quality_threshold,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"获取随机历史失败: {e}")
            return None

enhanced_ai = EnhancedAI()
# ============================

# ========== 核心插件逻辑 ==========
class LearningChatPlus:
    def __init__(self):
        # 注册事件处理器
        self.message_handler = on_message(priority=99, block=False)
        self.poke_handler = on_notice(rule=lambda event: isinstance(event, PokeNotifyEvent))
        
        self.message_handler.append_handler(self.handle_message)
        self.poke_handler.append_handler(self.handle_poke)
        
        # 冷却时间记录
        self.poke_last_time = {}
        self.msg_last_time = {}
        self.at_last_time = {}
        
        # 内容过滤列表
        self.prohibited_words = [
            "傻逼", "傻比", "sb", "SB", "cnm", "操你妈", "草你妈", "fuck", "Fuck", "shit", "Shit",
            "妈的", "妈逼", "狗东西", "狗比", "日你", "滚你", "你妈", "鸡巴", "你妹", "你爹", "你大爷",
            "去死", "变态", "神经病", "智障", "脑残", "弱智", "二货", "二逼", "二百五", "垃圾", "废物",
            "屌丝", "装逼", "逼样", "死开", "傻狗", "傻叉", "傻缺", "杂种", "贱人", "贱货", "婊子"
        ]
        
        # 特殊内容过滤 - 增加更多无法查看的消息类型
        self.special_content_patterns = [
            r"该消息.*不支持查看",
            r"不支持的消息类型",
            r"\[不支持的消息\]",
            r"\[图片\]",
            r"\[视频\]",
            r"\[语音\]",
            r"\[文件\]",
            r"\[动画表情\]",
            r"\[.*\]",
            r"该消息.*无法查看",   # 新增
            r"暂不支持的消息类型",  # 新增
            r"微信消息",          # 新增，如果包含微信消息
            r"QQ红包"            # 新增，QQ红包消息
            r"#"
        ]
    
    def is_inappropriate(self, text: str) -> bool:
        """检查文本是否包含不适当内容"""
        # 检查脏话
        if any(word in text for word in self.prohibited_words):
            return True
        
        # 检查特殊内容模式 - 匹配无法查看的消息
        if any(re.search(pattern, text) for pattern in self.special_content_patterns):
            return True
        
        # 检查命令消息
        if text.startswith('/'):
            return True
            
        # 检查消息长度是否过短
        if len(text.strip()) < config.min_message_length:
            return True
        
        # 检查违规词
        if config.blocked_words_enabled and any(word in text for word in config.blocked_words):
            return True
            
        return False
    
    async def is_group_enabled(self, group_id: int, feature: str) -> bool:
        """检查群组功能是否启用 - 增强稳定性"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                # 确保表存在
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='group_settings'"
                )
                if not cursor.fetchone():
                    # 表不存在，创建
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS group_settings (
                        group_id INTEGER PRIMARY KEY,
                        learning_enabled BOOLEAN DEFAULT 1,
                        poke_enabled BOOLEAN DEFAULT 1,
                        emoji_enabled BOOLEAN DEFAULT 1,
                        at_reply_enabled BOOLEAN DEFAULT 1,
                        ai_mode BOOLEAN DEFAULT 1,
                        speak_enabled BOOLEAN DEFAULT 1,
                        learn_enabled BOOLEAN DEFAULT 1
                    )
                    """)
                    conn.commit()
                    logger.info("创建缺失的group_settings表")
                
                # 检查表结构
                cursor.execute("PRAGMA table_info(group_settings)")
                columns = [row[1] for row in cursor.fetchall()]
                required_columns = ['learning_enabled', 'poke_enabled', 'emoji_enabled', 'at_reply_enabled', 'ai_mode', 'speak_enabled', 'learn_enabled']
                
                for col in required_columns:
                    if col not in columns:
                        logger.warning(f"检测到缺失列 {col}，尝试修复...")
                        cursor.execute(f"ALTER TABLE group_settings ADD COLUMN {col} BOOLEAN DEFAULT 1")
                        conn.commit()
                
                # 获取设置
                cursor.execute(
                    "SELECT learning_enabled, poke_enabled, emoji_enabled, at_reply_enabled, speak_enabled, learn_enabled FROM group_settings WHERE group_id=?",
                    (group_id,)
                )
                result = cursor.fetchone()
                
                # 如果没有记录，使用默认值
                if not result:
                    if feature == "learning":
                        return config.default_group_learn_enabled
                    elif feature == "poke":
                        return config.default_poke_enabled
                    elif feature == "emoji":
                        return config.default_emoji_enabled
                    elif feature == "at_reply":
                        return config.default_at_reply_enabled
                    elif feature == "speak":
                        return config.default_group_speak_enabled
                    elif feature == "learn":
                        return config.default_group_learn_enabled
                    else:
                        return True
                
                # 根据功能类型返回状态
                if feature == "learning":
                    return bool(result[0])
                elif feature == "poke":
                    return bool(result[1])
                elif feature == "emoji":
                    return bool(result[2])
                elif feature == "at_reply":
                    return bool(result[3])
                elif feature == "speak":
                    return bool(result[4])
                elif feature == "learn":
                    return bool(result[5])
                
                return True
        except Exception as e:
            logger.error(f"检查群组状态失败: {e}")
            # 返回默认值
            if feature == "learning":
                return config.default_group_learn_enabled
            elif feature == "poke":
                return config.default_poke_enabled
            elif feature == "emoji":
                return config.default_emoji_enabled
            elif feature == "at_reply":
                return config.default_at_reply_enabled
            elif feature == "speak":
                return config.default_group_speak_enabled
            elif feature == "learn":
                return config.default_group_learn_enabled
            return True
    
    def create_at_reply(self, user_id: int, message: str) -> Message:
        """创建正确的艾特回复消息"""
        if config.at_reply_format == 'cq':
            # 创建原生艾特格式
            return Message([
                MessageSegment.at(user_id),
                MessageSegment.text(" " + message)
            ])
        else:
            # 文本格式显示
            return Message(f"@{user_id} {message}")
    
    async def handle_message(self, bot: Bot, event: GroupMessageEvent):
        """处理群聊消息学习 - 全局数据管理"""
        try:
            # 排除机器人自身消息
            if event.user_id == bot.self_id:
                return
            
            # 检查群发言功能是否启用
            if not await self.is_group_enabled(event.group_id, "speak"):
                return
            
            # 获取原始消息文本
            raw_msg = event.get_plaintext().strip()
            
            # 检查消息长度
            if len(raw_msg) > config.max_message_length:
                return
                
            # 检查消息是否过短
            if len(raw_msg) < config.min_message_length:
                logger.info(f"忽略过短消息: {raw_msg}")
                return
                
            # 内容过滤 - 检查无法查看的消息
            if self.is_inappropriate(raw_msg):
                logger.info(f"忽略不适当消息: {raw_msg}")
                return
                
            # 评估消息质量
            quality_score = enhanced_ai.assess_message_quality(raw_msg)
            if quality_score < config.message_quality_threshold:
                logger.info(f"忽略低质量消息 (分数: {quality_score:.2f}): {raw_msg}")
                return
                
            # 检查冷却时间 (防止刷屏)
            current_time = time.time()
            last_time = self.msg_last_time.get(event.group_id, 0)
            if current_time - last_time < 2:  # 2秒冷却
                return
            self.msg_last_time[event.group_id] = current_time
            
            # 如果群学习功能启用，则保存消息到全局数据库
            if await self.is_group_enabled(event.group_id, "learn"):
                try:
                    with sqlite3.connect(config.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """INSERT INTO chat_logs (group_id, user_id, message, quality_score) 
                            VALUES (?, ?, ?, ?)""",
                            (event.group_id, event.user_id, raw_msg, quality_score)
                        )
                        message_id = cursor.lastrowid
                        conn.commit()
                        
                        # 缓存消息用于训练
                        enhanced_ai.message_cache.append(raw_msg)
                        
                        # 异步更新关键词关联
                        asyncio.create_task(enhanced_ai.update_keyword_associations(raw_msg, message_id, event.group_id, quality_score))
                except Exception as e:
                    logger.error(f"保存消息失败: {e}")
            
            # 更新上下文
            enhanced_ai.update_context(event.group_id, event.user_id, raw_msg)
            
            # ========== 艾特回复功能 ==========
            if event.is_tome() and await self.is_group_enabled(event.group_id, "at_reply"):
                # 冷却检查
                last_at_time = self.at_last_time.get(event.group_id, 0)
                if current_time - last_at_time < 5:  # 5秒冷却
                    return
                self.at_last_time[event.group_id] = current_time
                
                # 概率回复
                if random.random() < config.at_reply_probability:
                    # 使用增强AI生成回复
                    ai_reply = await enhanced_ai.generate_response(event.group_id, event.user_id, raw_msg)
                    
                    if ai_reply:
                        # 检查是否触发连续回复
                        if random.random() < config.sequential_reply_probability:
                            # 连续回复 (2-4条)
                            reply_count = min(random.randint(2, config.max_sequential_reply), config.max_sequential_reply)
                            messages = await self.get_sequential_messages(reply_count)
                            
                            if messages:
                                # 发送连续回复
                                for i, msg in enumerate(messages):
                                    delay = random.uniform(*config.sequential_reply_delay)
                                    await asyncio.sleep(delay)
                                    # 只在第一条消息添加艾特
                                    if config.at_reply_at_sender and i == 0:
                                        reply_msg = self.create_at_reply(event.user_id, msg)
                                    else:
                                        reply_msg = msg
                                    try:
                                        await bot.send(event, reply_msg)
                                    except ActionFailed:
                                        logger.error(f"发送消息失败，可能被风控: {msg[:20]}...")
                                    except Exception as e:
                                        logger.error(f"发送消息时发生未知错误: {e}")
                                logger.info(f"已发送连续回复: {reply_count}条")
                                return  # 避免触发其他回复
                        
                        # 单条回复
                        if config.at_reply_at_sender:
                            reply_msg = self.create_at_reply(event.user_id, ai_reply)
                        else:
                            reply_msg = ai_reply
                            
                        try:
                            await bot.send(event, reply_msg)
                            logger.info(f"已发送AI回复: {ai_reply[:20]}...")
                        except ActionFailed:
                            logger.error(f"发送消息失败，可能被风控: {ai_reply[:20]}...")
                        except Exception as e:
                            logger.error(f"发送消息时发生未知错误: {e}")
            # =====================================
            
            # 检查表情包功能是否启用
            if not await self.is_group_enabled(event.group_id, "emoji"):
                return
            
            # 概率触发表情包 (非回复机器人时)
            if not event.is_tome() and random.random() < config.emoji_probability:
                emoji_path = await self.get_random_emoji(event.group_id, raw_msg)
                if emoji_path:
                    try:
                        # 发送表情包
                        await bot.send(event, MessageSegment.image(emoji_path))
                        logger.info(f"已发送表情包: {os.path.basename(emoji_path)}")
                        
                        # 小概率追加AI回复
                        if random.random() < 0.3:
                            ai_reply = await enhanced_ai.generate_response(event.group_id, event.user_id, raw_msg)
                            if ai_reply:
                                try:
                                    await bot.send(event, ai_reply)
                                    logger.info(f"已发送表情包附加回复: {ai_reply[:20]}...")
                                except ActionFailed:
                                    logger.error(f"发送消息失败，可能被风控: {ai_reply[:20]}...")
                                except Exception as e:
                                    logger.error(f"发送消息时发生未知错误: {e}")
                    except Exception as e:
                        logger.error(f"发送表情包失败: {e}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
    
    async def handle_poke(self, bot: Bot, event: PokeNotifyEvent):
        """处理戳一戳事件 - 增强稳定性"""
        try:
            logger.info(f"收到戳一戳事件: {event.dict()}")
            
            # 仅处理群聊戳一戳
            if not hasattr(event, "group_id") or not event.group_id:
                logger.info("忽略非群聊戳一戳")
                return
            
            # 检查群发言功能是否启用
            if not await self.is_group_enabled(event.group_id, "speak"):
                logger.info(f"群 {event.group_id} 发言功能已禁用")
                return
            
            # 修复目标检查问题 - 确保类型一致
            target_id = str(event.target_id)
            self_id = str(bot.self_id)
            
            if target_id != self_id:
                logger.info(f"忽略非指向机器人的戳一戳 (目标: {target_id}, 机器人: {self_id})")
                return
                
            # 检查戳一戳功能是否启用
            if not await self.is_group_enabled(event.group_id, "poke"):
                logger.info(f"群 {event.group_id} 戳一戳功能已禁用")
                return
                
            # 跨群冷却检查
            current_time = time.time()
            last_time = self.poke_last_time.get(event.group_id, 0)
            if current_time - last_time < config.poke_cooldown:
                logger.info(f"戳一戳冷却中，剩余时间: {config.poke_cooldown - (current_time - last_time):.1f}秒")
                return
                
            # 确定目标用户
            target_id = event.user_id
            
            # 更新冷却时间
            self.poke_last_time[event.group_id] = current_time
            
            # 高概率戳回用户
            if random.random() < config.poke_back_probability:
                # 随机延迟戳回 (0.5-2秒)
                delay = random.uniform(*config.poke_back_delay)
                logger.info(f"将在 {delay:.1f} 秒后回应戳一戳")
                await asyncio.sleep(delay)
                
                # 使用文本代替戳一戳动作
                poke_texts = [
                    "戳回去！", "别戳我啦~", "反戳！", "戳一戳你！",
                    "我戳！", "戳戳戳！", "再戳我要生气了！", "戳回去！"
                ]
                
                # 发送文本戳回消息
                poke_msg = random.choice(poke_texts)
                try:
                    await bot.send(event, poke_msg)
                    logger.info(f"已发送文本戳回: {poke_msg}")
                except ActionFailed:
                    logger.error(f"发送消息失败，可能被风控: {poke_msg}")
                except Exception as e:
                    logger.error(f"发送消息时发生未知错误: {e}")
            
            # 使用增强AI生成回复
            ai_reply = await enhanced_ai.generate_response(event.group_id, target_id, "戳一戳")
            if ai_reply:
                # 在回复时艾特发送者
                if config.at_reply_at_sender:
                    reply_msg = self.create_at_reply(target_id, ai_reply)
                else:
                    reply_msg = ai_reply
                try:
                    await bot.send(event, reply_msg)
                    logger.info(f"已发送AI回复: {ai_reply[:20]}...")
                except ActionFailed:
                    logger.error(f"发送消息失败，可能被风控: {ai_reply[:20]}...")
                except Exception as e:
                    logger.error(f"发送消息时发生未知错误: {e}")
                return
                
            # 随机选择历史消息
            history_msg = await self.get_random_history()
            if history_msg:
                # 在回复时艾特发送者
                if config.at_reply_at_sender:
                    reply_msg = self.create_at_reply(target_id, history_msg)
                else:
                    reply_msg = history_msg
                try:
                    await bot.send(event, reply_msg)
                    logger.info(f"已发送历史消息: {history_msg[:20]}...")
                except ActionFailed:
                    logger.error(f"发送消息失败，可能被风控: {history_msg[:20]}...")
                except Exception as e:
                    logger.error(f"发送消息时发生未知错误: {e}")
                
                # 检查表情包功能是否启用
                if not await self.is_group_enabled(event.group_id, "emoji"):
                    return
                
                # 70%概率附加表情包
                if random.random() < 0.7:
                    emoji_path = await self.get_random_emoji(event.group_id)
                    if emoji_path:
                        try:
                            await bot.send(event, MessageSegment.image(emoji_path))
                            logger.info(f"已附加表情包: {os.path.basename(emoji_path)}")
                        except ActionFailed:
                            logger.error(f"发送表情包失败，可能被风控")
                        except Exception as e:
                            logger.error(f"发送表情包时发生未知错误: {e}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"处理戳一戳失败: {e}")
    
    async def get_random_history(self) -> Optional[str]:
        """从全局数据库随机获取历史消息"""
        try:
            return await enhanced_ai.get_random_history()
        except Exception as e:
            logger.error(f"获取历史消息失败: {e}")
            return None
    
    async def get_sequential_messages(self, count: int) -> List[str]:
        """
        获取按时间顺序连续的消息 - 全局数据
        :param count: 消息数量 (2-4)
        :return: 消息列表 (按时间顺序)
        """
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取最近消息的ID
                cursor.execute(
                    "SELECT id FROM chat_logs WHERE quality_score >= ? ORDER BY id DESC LIMIT 1000",
                    (config.message_quality_threshold,)
                )
                id_results = cursor.fetchall()
                
                if not id_results or len(id_results) < count:
                    return []
                
                # 随机选择一个起始位置
                start_index = random.randint(0, len(id_results) - count)
                # 获取连续的消息ID
                selected_ids = [row[0] for row in id_results[start_index:start_index+count]]
                
                # 按时间顺序获取消息内容 (ID从小到大)
                placeholders = ','.join(['?'] * len(selected_ids))
                cursor.execute(
                    f"SELECT message FROM chat_logs WHERE id IN ({placeholders}) AND quality_score >= ? ORDER BY id ASC",
                    selected_ids + [config.message_quality_threshold]
                )
                results = cursor.fetchall()
                return [row[0] for row in results] if results else []
        except Exception as e:
            logger.error(f"获取连续消息失败: {e}")
            return []
    
    async def get_random_emoji(self, group_id: int, trigger: str = None) -> Optional[str]:
        """根据触发词获取本地表情包 - 增强稳定性"""
        try:
            # 优先从数据库获取
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                # 优先匹配触发词
                if trigger:
                    # 提取关键词（限制最大数量）
                    keywords = re.findall(r'\w{2,}', trigger)[:config.max_keywords]
                    
                    if keywords:
                        # 构建查询条件（限制关键词数量）
                        conditions = []
                        params = []
                        for kw in keywords:
                            conditions.append("trigger_words LIKE ?")
                            params.append(f"%{kw}%")
                        
                        # 执行查询
                        query = " OR ".join(conditions)
                        cursor.execute(
                            f"""SELECT file_path FROM emojis 
                            WHERE (group_id=0 OR group_id=?) 
                            AND ({query}) 
                            ORDER BY RANDOM() LIMIT 1""",
                            (group_id, *params)
                        )
                        result = cursor.fetchone()
                        if result and os.path.exists(result[0]):
                            return result[0]
                
                # 随机选择全局表情
                cursor.execute(
                    """SELECT file_path FROM emojis 
                    WHERE group_id=0 OR group_id=?
                    ORDER BY RANDOM() LIMIT 1""",
                    (group_id,)
                )
                result = cursor.fetchone()
                if result and os.path.exists(result[0]):
                    return result[0]
            
            # 如果数据库没有，从本地文件夹随机选择
            return await self.get_random_emoji_from_folder()
                
        except Exception as e:
            logger.error(f"获取表情包失败: {e}")
            return await self.get_random_emoji_from_folder()
    
    async def get_random_emoji_from_folder(self) -> Optional[str]:
        """从本地文件夹随机获取表情包 - 增强稳定性"""
        try:
            if not config.emoji_path.exists():
                config.emoji_path.mkdir(parents=True, exist_ok=True)
                return None
                
            emoji_files = [
                f for f in os.listdir(config.emoji_path) 
                if os.path.splitext(f)[1].lower() in config.emoji_formats
            ]
            if emoji_files:
                file_path = config.emoji_path / random.choice(emoji_files)
                logger.info(f"从文件夹选择表情包: {file_path.name}")
                return str(file_path)
            return None
        except Exception as e:
            logger.error(f"从文件夹获取表情包失败: {e}")
            return None

# ========== 群管理命令 ==========
# 定义群管理员和群主的权限
GROUP_MANAGER = GROUP_ADMIN | GROUP_OWNER | SUPERUSER

# 开关命令
switch_learning = on_command("学习开关", aliases={"学习功能"}, permission=GROUP_MANAGER, priority=1)
@switch_learning.handle()
async def switch_learning_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制学习功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "学习功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "学习功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "learning")
        try:
            await switch_learning.finish(f"学习功能当前状态: {'开启' if current else '关闭'}\n用法: /学习开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, learning_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 学习功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_learning.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置学习功能状态失败: {e}")
        try:
            await switch_learning.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

switch_poke = on_command("戳一戳开关", aliases={"戳戳功能"}, permission=GROUP_MANAGER, priority=1)
@switch_poke.handle()
async def switch_poke_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制戳一戳功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "戳一戳功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "戳一戳功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "poke")
        try:
            await switch_poke.finish(f"戳一戳功能当前状态: {'开启' if current else '关闭'}\n用法: /戳一戳开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, poke_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 戳一戳功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_poke.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置戳一戳功能状态失败: {e}")
        try:
            await switch_poke.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

switch_emoji = on_command("表情开关", aliases={"表情功能"}, permission=GROUP_MANAGER, priority=1)
@switch_emoji.handle()
async def switch_emoji_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制表情功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "表情功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "表情功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "emoji")
        try:
            await switch_emoji.finish(f"表情功能当前状态: {'开启' if current else '关闭'}\n用法: /表情开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, emoji_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 表情功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_emoji.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置表情功能状态失败: {e}")
        try:
            await switch_emoji.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 新增：艾特回复开关
switch_at_reply = on_command("艾特回复开关", aliases={"艾特回复功能"}, permission=GROUP_MANAGER, priority=1)
@switch_at_reply.handle()
async def switch_at_reply_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制艾特回复功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "艾特回复功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "艾特回复功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "at_reply")
        try:
            await switch_at_reply.finish(f"艾特回复功能当前状态: {'开启' if current else '关闭'}\n用法: /艾特回复开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, at_reply_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 艾特回复功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_at_reply.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置艾特回复功能状态失败: {e}")
        try:
            await switch_at_reply.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# AI模式开关
switch_ai_mode = on_command("AI模式", aliases={"智能模式"}, permission=GROUP_MANAGER, priority=1)
@switch_ai_mode.handle()
async def switch_ai_mode_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制AI模式开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "AI模式已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "AI模式已关闭"
    else:
        # 查询当前状态
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT ai_mode FROM group_settings WHERE group_id=?",
                    (group_id,)
                )
                result = cursor.fetchone()
                current = result[0] if result else True
                try:
                    await switch_ai_mode.finish(f"AI模式当前状态: {'开启' if current else '关闭'}\n用法: /AI模式 [开/关]")
                except FinishedException:
                    return
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    return
        except Exception as e:
            logger.error(f"查询AI模式失败: {e}")
            try:
                await switch_ai_mode.finish("查询状态失败")
            except FinishedException:
                return
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, ai_mode) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        try:
            await switch_ai_mode.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置AI模式失败: {e}")
        try:
            await switch_ai_mode.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 新增：发言开关
switch_speak = on_command("发言开关", aliases={"发言功能"}, permission=GROUP_MANAGER, priority=1)
@switch_speak.handle()
async def switch_speak_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制发言功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "发言功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "发言功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "speak")
        try:
            await switch_speak.finish(f"发言功能当前状态: {'开启' if current else '关闭'}\n用法: /发言开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, speak_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 发言功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_speak.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置发言功能状态失败: {e}")
        try:
            await switch_speak.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 新增：群学习开关
switch_group_learn = on_command("群学习开关", aliases={"群学习功能"}, permission=GROUP_MANAGER, priority=1)
@switch_group_learn.handle()
async def switch_group_learn_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制群学习功能开关"""
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id
    
    if arg in ["开", "开启", "on", "enable"]:
        enabled = 1
        msg = "群学习功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        enabled = 0
        msg = "群学习功能已关闭"
    else:
        # 查询当前状态
        current = await plugin.is_group_enabled(group_id, "learn")
        try:
            await switch_group_learn.finish(f"群学习功能当前状态: {'开启' if current else '关闭'}\n用法: /群学习开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 更新或插入群组设置
            cursor.execute(
                """INSERT OR REPLACE INTO group_settings (group_id, learn_enabled) 
                VALUES (?, ?)""",
                (group_id, enabled)
            )
            conn.commit()
        logger.info(f"群 {group_id} 学习功能状态: {'开启' if enabled else '关闭'}")
        try:
            await switch_group_learn.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置群学习功能状态失败: {e}")
        try:
            await switch_group_learn.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 逻辑思维增强开关
switch_logical_thinking = on_command("逻辑增强", aliases={"逻辑思维"}, permission=GROUP_MANAGER, priority=1)
@switch_logical_thinking.handle()
async def switch_logical_thinking_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制逻辑思维增强开关"""
    arg = args.extract_plain_text().strip().lower()
    
    if arg in ["开", "开启", "on", "enable"]:
        config.logical_thinking_enabled = 1
        msg = "逻辑思维增强已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        config.logical_thinking_enabled = 0
        msg = "逻辑思维增强已关闭"
    else:
        # 查询当前状态
        current = config.logical_thinking_enabled
        try:
            await switch_logical_thinking.finish(f"逻辑思维增强当前状态: {'开启' if current else '关闭'}\n用法: /逻辑增强 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    # 保存配置
    try:
        config_parser = configparser.ConfigParser()
        config_parser.read(config.config_file, encoding='utf-8')
        config_parser.set('DEFAULT', 'logical_thinking_enabled', str(config.logical_thinking_enabled))
        
        with open(config.config_file, 'w', encoding='utf-8') as f:
            config_parser.write(f)
        
        logger.info(f"逻辑思维增强状态: {'开启' if config.logical_thinking_enabled else '关闭'}")
        try:
            await switch_logical_thinking.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置逻辑思维增强状态失败: {e}")
        try:
            await switch_logical_thinking.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 删除学习内容 - 增强稳定性
delete_learning = on_command("删除学习", aliases={"忘记学习"}, permission=GROUP_MANAGER, priority=1)
@delete_learning.handle()
async def delete_learning_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """删除学习内容 - 全局数据"""
    keyword = args.extract_plain_text().strip()
    
    if not keyword:
        try:
            await delete_learning.finish("请提供要删除的关键词\n用法: /删除学习 [关键词]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        total_deleted = 0
        batch_size = 100  # 每批删除100条
        
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            
            # 分批删除，避免SQL变量过多问题
            while True:
                cursor.execute(
                    "DELETE FROM chat_logs WHERE id IN ("
                    "   SELECT id FROM chat_logs "
                    "   WHERE message LIKE ? "
                    "   LIMIT ?"
                    ")",
                    (f"%{keyword}%", batch_size)
                )
                deleted_count = cursor.rowcount
                total_deleted += deleted_count
                conn.commit()
                
                if deleted_count < batch_size:
                    break
        
        if total_deleted > 0:
            msg = f"已删除 {total_deleted} 条包含 '{keyword}' 的学习内容"
        else:
            msg = f"未找到包含 '{keyword}' 的学习内容"
        
        logger.info(f"{msg}")
        try:
            await delete_learning.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"删除学习内容失败: {e}")
        try:
            await delete_learning.finish("删除失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 查看学习内容
list_learning = on_command("学习列表", permission=GROUP_MANAGER, priority=1)
@list_learning.handle()
async def list_learning_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """查看全局学习内容"""
    keyword = args.extract_plain_text().strip()
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            
            if keyword:
                # 搜索匹配关键词的学习内容
                cursor.execute(
                    "SELECT id, message, quality_score FROM chat_logs WHERE message LIKE ? LIMIT 10",
                    (f"%{keyword}%",)
                )
                results = cursor.fetchall()
                count = len(results)
                title = f"找到 {count} 条包含 '{keyword}' 的学习内容:\n"
            else:
                # 获取最新的学习内容
                cursor.execute(
                    "SELECT id, message, quality_score FROM chat_logs ORDER BY id DESC LIMIT 10"
                )
                results = cursor.fetchall()
                count = len(results)
                title = f"最近 {count} 条学习内容:\n"
            
            if not results:
                try:
                    await list_learning.finish("没有找到学习内容")
                except FinishedException:
                    return
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    return
            
            msg = title
            for idx, (msg_id, message, quality) in enumerate(results, 1):
                quality_str = f"[质量:{quality:.2f}]" if quality < 0.9 else ""
                msg += f"{idx}. [ID:{msg_id}] {quality_str} {message[:30]}{'...' if len(message) > 30 else ''}\n"
            
            if keyword:
                msg += "\n提示: 使用 /删除学习 [关键词] 删除内容"
            else:
                msg += "\n提示: 使用 /删除学习 [关键词] 删除匹配内容"
            
            try:
                await list_learning.finish(msg)
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"获取学习列表失败: {e}")
        try:
            await list_learning.finish("获取列表失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 新增功能：删除空消息和全部消息
delete_empty = on_command("删除空消息", aliases={"清空空消息"}, permission=GROUP_MANAGER, priority=1)
@delete_empty.handle()
async def delete_empty_handler(bot: Bot, event: GroupMessageEvent):
    """删除所有空字符消息 - 全局"""
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 删除空消息：message为空字符串或空白字符
            cursor.execute(
                "DELETE FROM chat_logs WHERE trim(message) = '' OR message IS NULL"
            )
            deleted = cursor.rowcount
            conn.commit()
            try:
                await delete_empty.finish(f"已删除 {deleted} 条空消息")
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"删除空消息失败: {e}")
        try:
            await delete_empty.finish("删除空消息失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

delete_all = on_command("删除所有学习", aliases={"清空学习"}, permission=GROUP_MANAGER, priority=1)
@delete_all.handle()
async def delete_all_handler(bot: Bot, event: GroupMessageEvent):
    """删除全局所有学习消息"""
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 删除所有消息
            cursor.execute("DELETE FROM chat_logs")
            deleted = cursor.rowcount
            conn.commit()
            
            # 重置AI模型
            enhanced_ai.vectorizer = None
            enhanced_ai.model_trained = False
            enhanced_ai.message_cache.clear()
            
            try:
                await delete_all.finish(f"已删除 {deleted} 条学习消息，AI模型已重置")
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"删除所有学习失败: {e}")
        try:
            await delete_all.finish("删除所有学习失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ========== 违规词管理命令 ==========
add_blocked_word = on_command("添加违规词", aliases={"违规词添加"}, permission=GROUP_MANAGER, priority=1)
@add_blocked_word.handle()
async def add_blocked_word_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """添加违规词"""
    word = args.extract_plain_text().strip()
    
    if not word:
        try:
            await add_blocked_word.finish("请提供要屏蔽的词语\n用法: /添加违规词 [词语]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    if word in config.blocked_words:
        try:
            await add_blocked_word.finish(f"'{word}' 已经在违规词列表中")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        # 添加到内存
        config.blocked_words.add(word)
        
        # 保存到文件
        config.save_blocked_words()
        
        # 添加到数据库
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO blocked_words (word) VALUES (?)",
                (word,)
            )
            conn.commit()
        
        try:
            await add_blocked_word.finish(f"已添加违规词: {word}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"添加违规词失败: {e}")
        try:
            await add_blocked_word.finish("添加违规词失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

remove_blocked_word = on_command("删除违规词", aliases={"违规词删除"}, permission=GROUP_MANAGER, priority=1)
@remove_blocked_word.handle()
async def remove_blocked_word_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """删除违规词"""
    word = args.extract_plain_text().strip()
    
    if not word:
        try:
            await remove_blocked_word.finish("请提供要删除的违规词\n用法: /删除违规词 [词语]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    if word not in config.blocked_words:
        try:
            await remove_blocked_word.finish(f"'{word}' 不在违规词列表中")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        # 从内存删除
        config.blocked_words.discard(word)
        
        # 保存到文件
        config.save_blocked_words()
        
        # 从数据库删除
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM blocked_words WHERE word = ?",
                (word,)
            )
            conn.commit()
        
        try:
            await remove_blocked_word.finish(f"已删除违规词: {word}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"删除违规词失败: {e}")
        try:
            await remove_blocked_word.finish("删除违规词失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

list_blocked_words = on_command("违规词列表", aliases={"查看违规词"}, permission=GROUP_MANAGER, priority=1)
@list_blocked_words.handle()
async def list_blocked_words_handler(bot: Bot, event: GroupMessageEvent):
    """查看违规词列表"""
    if not config.blocked_words:
        try:
            await list_blocked_words.finish("当前没有设置任何违规词")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    words = list(config.blocked_words)
    words.sort()
    
    # 分页显示
    page_size = 20
    pages = [words[i:i+page_size] for i in range(0, len(words), page_size)]
    
    if not pages:
        try:
            await list_blocked_words.finish("当前没有设置任何违规词")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    msg = f"违规词列表 (共 {len(words)} 个):\n"
    for i, word in enumerate(pages[0], 1):
        msg += f"{i}. {word}\n"
    
    if len(pages) > 1:
        msg += f"\n第1页，共{len(pages)}页，使用 /违规词列表 [页码] 查看其他页"
    
    try:
        await list_blocked_words.finish(msg)
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"发送消息失败: {e}")

switch_blocked_words = on_command("违规词开关", aliases={"违规词功能"}, permission=GROUP_MANAGER, priority=1)
@switch_blocked_words.handle()
async def switch_blocked_words_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """控制违规词功能开关"""
    arg = args.extract_plain_text().strip().lower()
    
    if arg in ["开", "开启", "on", "enable"]:
        config.blocked_words_enabled = 1
        msg = "违规词功能已开启"
    elif arg in ["关", "关闭", "off", "disable"]:
        config.blocked_words_enabled = 0
        msg = "违规词功能已关闭"
    else:
        # 查询当前状态
        current = config.blocked_words_enabled
        try:
            await switch_blocked_words.finish(f"违规词功能当前状态: {'开启' if current else '关闭'}\n用法: /违规词开关 [开/关]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    # 保存配置
    try:
        config_parser = configparser.ConfigParser()
        config_parser.read(config.config_file, encoding='utf-8')
        config_parser.set('DEFAULT', 'blocked_words_enabled', str(config.blocked_words_enabled))
        
        with open(config.config_file, 'w', encoding='utf-8') as f:
            config_parser.write(f)
        
        logger.info(f"违规词功能状态: {'开启' if config.blocked_words_enabled else '关闭'}")
        try:
            await switch_blocked_words.finish(msg)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"设置违规词功能状态失败: {e}")
        try:
            await switch_blocked_words.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ========== 表情包管理命令 ==========
add_emoji = on_command("添加表情", aliases={"添加表情包"}, permission=GROUP_MANAGER, priority=1)
@add_emoji.handle()
async def add_emoji_handler(bot: Bot, event: GroupMessageEvent):
    """添加本地表情包到数据库"""
    # 检查群表情功能是否启用
    if not await plugin.is_group_enabled(event.group_id, "emoji"):
        try:
            await add_emoji.finish("本群表情功能已禁用")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    # 检查消息中是否包含图片
    image_segs = [seg for seg in event.message if seg.type == "image"]
    
    if not image_segs:
        try:
            await add_emoji.finish("未检测到图片，请附带图片发送命令")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    # 获取触发词（去除命令部分）
    trigger_words = event.get_plaintext().replace("添加表情", "").replace("添加表情包", "").strip()
    
    success_count = 0
    for seg in image_segs:
        url = seg.data.get("url", "")
        if not url:
            continue
            
        try:
            # 下载图片
            local_path = await download_image(url, config.emoji_path)
            logger.info(f"已下载图片: {local_path.name}")
            
            # 验证图片有效性
            if not await validate_image(local_path):
                os.remove(local_path)
                continue
                
            # 保存到数据库
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR IGNORE INTO emojis 
                    (group_id, file_path, trigger_words) 
                    VALUES (?, ?, ?)""",
                    (event.group_id, str(local_path), trigger_words)
                )
                conn.commit()
                success_count += 1
                logger.info(f"已添加表情包到数据库: {local_path.name}")
        except Exception as e:
            logger.error(f"添加表情包失败: {e}")
    
    try:
        await add_emoji.finish(f"成功添加 {success_count}/{len(image_segs)} 个表情包！")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"发送消息失败: {e}")

list_emojis = on_command("表情列表", permission=GROUP_MANAGER, priority=1)
@list_emojis.handle()
async def list_emojis_handler(bot: Bot, event: GroupMessageEvent):
    """查看表情包列表"""
    # 检查群表情功能是否启用
    if not await plugin.is_group_enabled(event.group_id, "emoji"):
        try:
            await list_emojis.finish("本群表情功能已禁用")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM emojis WHERE group_id=0 OR group_id=?",
                (event.group_id,)
            )
            count = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT id, file_path, trigger_words FROM emojis WHERE group_id=0 OR group_id=? LIMIT 5",
                (event.group_id,)
                )
            results = cursor.fetchall()
            
            msg = f"本群共有 {count} 个表情包\n最新添加的5个：\n"
            for emoji_id, path, triggers in results:
                filename = os.path.bassubpath(path)
                msg += f"[ID:{emoji_id}] {filename} - 触发词: {triggers or '无'}\n"
            
            msg += "\n提示: 使用 /删除表情 [ID] 删除表情包"
            try:
                await list_emojis.finish(msg)
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"获取表情列表失败: {e}")
        try:
            await list_emojis.finish("获取表情列表失败")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

delete_emoji = on_command("删除表情", permission=GROUP_MANAGER, priority=1)
@delete_emoji.handle()
async def delete_emoji_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """删除表情包"""
    emoji_id = args.extract_plain_text().strip()
    group_id = event.group_id
    
    if not emoji_id.isdigit():
        try:
            await delete_emoji.finish("请提供有效的表情ID\n用法: /删除表情 [ID]\n使用 /表情列表 查看ID")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        emoji_id = int(emo_id)
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            # 查询表情包路径
            cursor.execute(
                "SELECT file_path FROM emojis WHERE id=? AND (group_id=0 OR group_id=?)",
                (emoji_id, group_id)
            )
            result = cursor.fetchone()
            
            if not result:
                try:
                    await delete_emoji.finish(f"未找到ID为 {emoji_id} 的表情包")
                except FinishedException:
                    return
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    return
            
            file_path = result[0]
            # 删除数据库记录
            cursor.execute(
                "DELETE FROM emojis WHERE id=?",
                (emoji_id,)
            )
            deleted = cursor.rowcount
            conn.commit()
            
            # 删除文件
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"已删除表情文件: {file_path}")
            
            try:
                await delete_emoji.finish(f"已删除表情包 ID: {emoji_id}")
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"删除表情包失败: {e}")
        try:
            await delete_emoji.finish("删除失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ========== 其他管理命令 ==========
poke_back_setting = on_command("戳回设置", permission=GROUP_MANAGER, priority=1)
@poke_back_setting.handle()
async def poke_back_setting_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """设置戳回参数"""
    arg_str = args.extract_plain_text().strip()
    if not arg_str:
        try:
            await poke_back_setting.finish("格式错误，请使用: /戳回设置 [概率] [最小延迟] [最大延迟]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    args = arg_str.split()
    
    try:
        # 更新戳回概率
        if len(args) > 0:
            new_prob = float(args[0])
            if 0 <= new_prob <= 1:
                config.poke_back_probability = new_prob
            else:
                raise ValueError("概率必须在0-1之间")
        
        # 更新延迟范围
        if len(args) > 2:
            min_delay = float(args[1])
            max_delay = float(args[2])
            if min_delay >= 0 and max_delay > min_delay:
                config.poke_back_delay_min = min_delay
                config.poke_back_delay_max = max_delay
                config.poke_back_delay = (min_delay, max_delay)
            else:
                raise ValueError("延迟范围无效")
        
        # 返回当前设置
        current_settings = (
            f"当前戳回设置:\n"
            f"戳回概率: {config.poke_back_probability*100:.1f}%\n"
            f"延迟范围: {config.poke_back_delay_min:.1f}-{config.poke_back_delay_max:.1f}秒"
        )
        logger.info(f"更新戳回设置: {current_settings}")
        try:
            await poke_back_setting.finish(current_settings)
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    except ValueError as e:
        logger.error(f"戳回设置失败: {e}")
        try:
            await poke_back_setting.finish(f"设置失败: {str(e)}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"戳回设置失败: {e}")
        try:
            await poke_back_setting.finish("设置失败，请重试")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

test_poke = on_command("测试戳回", permission=GROUP_MANAGER, priority=1)
@test_poke.handle()
async def test_poke_handler(bot: Bot, event: GroupMessageEvent):
    """测试戳回功能"""
    try:
        # 模拟戳一戳事件
        class MockPokeEvent(PokeNotifyEvent):
            self_id = bot.self_id
            time = int(time.time())
            user_id = event.user_id
            group_id = event.group_id
            target_id = bot.self_id
            
        mock_event = MockPokeEvent(
            time=time.time(),
            self_id=bot.self_id,
            post_type="notice",
            notice_type="notify",
            sub_type="poke",
            user_id=event.user_id,
            target_id=bot.self_id,
            group_id=event.group_id
        )
        
        # 处理模拟事件
        await plugin.handle_poke(bot, mock_event)
        try:
            await test_poke.finish("已触发测试戳回，请检查效果")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"测试戳回失败: {e}")
        try:
            await test_poke.finish(f"测试失败: {str(e)}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 新增：测试艾特回复命令
test_at_reply = on_command("测试艾特回复", permission=GROUP_MANAGER, priority=1)
@test_at_reply.handle()
async def test_at_reply_handler(bot: Bot, event: GroupMessageEvent):
    """测试艾特回复功能"""
    try:
        # 模拟艾特事件
        class MockAtEvent(GroupMessageEvent):
            message = Message(f"[CQ:at,qq={bot.self_id}] 测试")
            self_id = bot.self_id
            time = int(time.time())
            user_id = event.user_id
            group_id = event.group_id
            message_id = "test_at_reply"
            to_me = True
            
            def is_tome(self):
                return True
                
            def get_plaintext(self):
                return "测试"
        
        mock_event = MockAtEvent(
            time=time.time(),
            self_id=bot.self_id,
            post_type="message",
            message_type="group",
            sub_type="normal",
            message_id="test_at_reply",
            user_id=event.user_id,
            group_id=event.group_id,
            message=Message(f"[CQ:at,qq={bot.self_id}] 测试"),
            raw_message=f"[CQ:at,qq={bot.self_id}] 测试",
            font=0,
            sender={"user_id": event.user_id, "nickname": "测试用户"}
        )
        
        # 处理模拟事件
        await plugin.handle_message(bot, mock_event)
        try:
            await test_at_reply.finish("已触发测试艾特回复，请检查效果")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"测试艾特回复失败: {e}")
        try:
            await test_at_reply.finish(f"测试失败: {str(e)}")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 查看关键词
view_keywords = on_command("查看关键词", permission=GROUP_MANAGER, priority=1)
@view_keywords.handle()
async def view_keywords_handler(bot: Bot, event: GroupMessageEvent):
    """查看全局关键词统计"""
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT keyword, COUNT(*) as count 
                FROM keyword_associations 
                GROUP BY keyword 
                ORDER BY count DESC LIMIT 10"""
            )
            results = cursor.fetchall()
            
            if not results:
                try:
                    await view_keywords.finish("尚未分析出关键词")
                except FinishedException:
                    return
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    return
            
            msg = "关键词统计 (TOP10):\n"
            for i, (keyword, count) in enumerate(results, 1):
                msg += f"{i}. {keyword}: {count}次关联\n"
            
            try:
                await view_keywords.finish(msg)
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"查看关键词失败: {e}")
        try:
            await view_keywords.finish("获取关键词失败")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# 查看关键词关联
view_keyword_associations = on_command("关键词关联", permission=GROUP_MANAGER, priority=1)
@view_keyword_associations.handle()
async def view_keyword_associations_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """查看关键词关联"""
    keyword = args.extract_plain_text().strip()
    
    if not keyword:
        try:
            await view_keyword_associations.finish("请指定关键词\n用法: /关键词关联 [关键词]")
        except FinishedException:
            return
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT associated_keyword, weight 
                FROM keyword_associations 
                WHERE keyword=?
                ORDER BY weight DESC LIMIT 5""",
                (keyword,)
            )
            results = cursor.fetchall()
            
            if not results:
                try:
                    await view_keyword_associations.finish(f"关键词 '{keyword}' 没有关联记录")
                except FinishedException:
                    return
                except Exception as e:
                    logger.error(f"发送消息失败: {e}")
                    return
            
            msg = f"关键词 '{keyword}' 的关联词:\n"
            for i, (assoc_keyword, weight) in enumerate(results, 1):
                msg += f"{i}. {assoc_keyword} (强度: {weight:.2f})\n"
            
            try:
                await view_keyword_associations.finish(msg)
            except FinishedException:
                pass
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    except Exception as e:
        logger.error(f"查看关键词关联失败: {e}")
        try:
            await view_keyword_associations.finish("获取关联失败")
        except FinishedException:
            pass
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ========== 重新训练AI功能 ==========
retrain_ai = on_command("训练AI", permission=SUPERUSER, priority=1)
@retrain_ai.handle()
async def retrain_ai_handler(bot: Bot, event: GroupMessageEvent):
    """重新训练AI模型"""
    try:
        # 先发送开始消息
        try:
            await bot.send(event, "开始重新训练AI模型，这可能需要一些时间...")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
        
        # 在后台执行训练任务
        asyncio.create_task(async_retrain_ai(bot, event))
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"启动训练任务失败: {e}")
        try:
            await bot.send(event, "训练启动失败，请查看日志")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

async def async_retrain_ai(bot: Bot, event: GroupMessageEvent):
    """异步执行AI训练"""
    try:
        # 执行训练
        await enhanced_ai.train_vectorizer()
        # 训练完成后发送消息
        try:
            await bot.send(event, "AI模型训练完成！")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"训练AI失败: {e}")
        try:
            await bot.send(event, "训练失败，请查看日志")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

# ========== 实用函数 ==========
async def download_image(url: str, save_dir: Path) -> Path:
    """下载图片到本地并返回路径 - 增强稳定性"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise ValueError(f"HTTP错误: {resp.status_code}")
            
            # 生成唯一文件名
            file_ext = get_image_extension(resp.content)
            if not file_ext:
                raise ValueError("不支持的图片格式")
                
            file_hash = hashlib.md5(resp.content).hexdigest()[:8]
            file_name = f"{int(time.time())}_{file_hash}{file_ext}"
            local_path = save_dir / file_name
            
            # 确保目录存在
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(resp.content)
                
            return local_path
    except Exception as e:
        logger.error(f"下载图片失败: {e}")
        raise

def get_image_extension(content: bytes) -> Optional[str]:
    """通过文件头识别图片格式"""
    if content.startswith(b'\xFF\xD8\xFF'):
        return '.jpg'
    elif content.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    elif content.startswith(b'GIF87a') or content.startswith(b'GIF89a'):
        return '.gif'
    elif content.startswith(b'RIFF') and content[8:12] == b'WEBP':
        return '.webp'
    return None

async def validate_image(file_path: Path) -> bool:
    """验证图片有效性并转换为标准格式 - 增强稳定性"""
    try:
        # 使用PIL打开图片
        with Image.open(file_path) as img:
            # 如果是GIF，直接返回（保留动画）
            if img.format == 'GIF':
                return True
                
            # 转换模式为RGB（兼容透明度）
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
                
            # 保存为JPEG格式
            new_path = file_path.with_suffix('.jpg')
            img.save(new_path, "JPEG", quality=85)
            
            # 删除原始文件（如果是其他格式）
            if file_path.suffix.lower() != '.jpg':
                os.remove(file_path)
                
            return True
    except Exception as e:
        logger.error(f"图片验证失败: {e}")
        if file_path.exists():
            try:
                os.remove(file_path)
            except:
                pass
        return False

# ========== 插件注册 ==========
plugin = LearningChatPlus()