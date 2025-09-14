import nonebot
from nonebot.adapters.onebot.v11 import (
    Bot,
    MessageEvent,
    GroupMessageEvent,
    Message,
    MessageSegment,
    GROUP_ADMIN,
    GROUP_OWNER
)
from nonebot.plugin import PluginMetadata, on_message, on_command
from nonebot.permission import SUPERUSER
from nonebot.params import CommandArg
from nonebot.exception import FinishedException
import sqlite3
import random
import time
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Deque, Optional
import jieba
import numpy as np
from collections import deque
import httpx
import aiofiles
from PIL import Image
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 导入主插件的核心功能
from .main_plugin import config, enhanced_ai, plugin as main_plugin

__plugin_meta__ = PluginMetadata(
    name="智能主动发言助手",
    description="让机器人主动发言、随机艾特群员和智能回复消息",
    usage="使用命令管理主动发言功能",
    extra={"author": "YourName", "version": "1.0"}
)

# 获取主插件的日志器
logger = config.logger

# ========== 分群管理设置 ==========
class GroupSettings:
    def __init__(self):
        # 群组设置: {group_id: {"active_enabled": bool, "at_enabled": bool}}
        self.group_settings: Dict[int, Dict[str, bool]] = {}
        self.db_path = Path("data/active_chat_settings.db")
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS group_settings (
                    group_id INTEGER PRIMARY KEY,
                    active_enabled INTEGER DEFAULT 1,
                    at_enabled INTEGER DEFAULT 1
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("主动发言设置数据库初始化成功")
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
    
    def load_settings(self):
        """从数据库加载所有群设置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT group_id, active_enabled, at_enabled FROM group_settings")
            rows = cursor.fetchall()
            for row in rows:
                group_id, active_enabled, at_enabled = row
                self.group_settings[group_id] = {
                    "active_enabled": bool(active_enabled),
                    "at_enabled": bool(at_enabled)
                }
            conn.close()
        except Exception as e:
            logger.error(f"加载群设置失败: {e}")
    
    def save_settings(self, group_id: int):
        """保存群设置到数据库"""
        try:
            if group_id not in self.group_settings:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            settings = self.group_settings[group_id]
            cursor.execute('''
                INSERT OR REPLACE INTO group_settings (group_id, active_enabled, at_enabled)
                VALUES (?, ?, ?)
            ''', (group_id, int(settings["active_enabled"]), int(settings["at_enabled"])))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存群设置失败: {e}")
    
    def get_group_setting(self, group_id: int, setting_key: str, default: bool = True) -> bool:
        """获取群设置"""
        if group_id not in self.group_settings:
            # 如果群设置不存在，使用默认值并保存
            self.group_settings[group_id] = {
                "active_enabled": True,
                "at_enabled": True
            }
            self.save_settings(group_id)
        
        return self.group_settings[group_id].get(setting_key, default)
    
    def set_group_setting(self, group_id: int, setting_key: str, value: bool):
        """设置群设置"""
        if group_id not in self.group_settings:
            self.group_settings[group_id] = {
                "active_enabled": True,
                "at_enabled": True
            }
        
        self.group_settings[group_id][setting_key] = value
        self.save_settings(group_id)

# 初始化群设置管理
group_settings_manager = GroupSettings()
group_settings_manager.load_settings()

# ========== 主动发言系统 ==========
class ActiveSpeaker:
    def __init__(self):
        # 活跃群组追踪: {group_id: (last_active_time, message_count)}
        self.active_groups: Dict[int, Tuple[float, int]] = {}
        # 消息冷却: {group_id: last_message_time}
        self.group_cooldowns: Dict[int, float] = {}
        # 用户发言追踪: {group_id: {user_id: deque}}
        self.user_activity: Dict[int, Dict[int, Deque[float]]] = {}
        # 用户最后发言时间: {group_id: {user_id: last_time}}
        self.user_last_active: Dict[int, Dict[int, float]] = {}
        # 主动发言概率配置
        self.active_speaker_probability = 0.2  # 基础触发概率
        self.max_cooldown = 60  # 最大冷却时间(秒)
        self.min_cooldown = 120  # 最小冷却时间(秒)
        self.min_activity = 5  # 触发主动发言的最小消息数
        self.mention_probability = 0.6  # 主动发言时艾特用户的概率
        self.context_window = 3  # 用户上下文窗口大小
        
        # 增加群聊氛围相关配置
        self.vibe_adjustment = True  # 是否根据群聊氛围调整发言
        self.emoji_probability = 0.5  # 发言中添加表情的概率
        self.meme_probability = 0.3  # 发送梗图的概率
        self.group_voice_patterns = {}  # 存储不同群的语言风格特征
        
        # 活跃用户互动策略
        self.interact_with_newcomers = True  # 是否主动与新成员互动
        self.include_inactive_users = 0.2  # 有时也与不活跃用户互动的概率
        self.follow_topic = True  # 是否跟随当前讨论主题
    
    def update_group_activity(self, group_id: int):
        """更新群组活跃度"""
        current_time = time.time()
        _, count = self.active_groups.get(group_id, (0.0, 0))
        self.active_groups[group_id] = (current_time, count + 1)
        
        # 初始化用户活动追踪
        if group_id not in self.user_activity:
            self.user_activity[group_id] = {}
            self.user_last_active[group_id] = {}
    
    def update_user_activity(self, group_id: int, user_id: int):
        """更新用户活跃度"""
        current_time = time.time()
        self.user_last_active[group_id][user_id] = current_time
        
        if user_id not in self.user_activity[group_id]:
            self.user_activity[group_id][user_id] = deque(maxlen=self.context_window)
        
        self.user_activity[group_id][user_id].append(current_time)
    
    def get_active_users(self, group_id: int) -> List[int]:
        """获取活跃用户列表(最近2分钟发言过的用户)"""
        current_time = time.time()
        active_users = []
        
        if group_id in self.user_last_active:
            for user_id, last_time in self.user_last_active[group_id].items():
                if current_time - last_time < 120:  # 最近2分钟活跃
                    active_users.append(user_id)
        
        return active_users
    
    def get_group_cooldown(self, group_id: int) -> float:
        """根据群活跃度计算冷却时间"""
        if group_id in self.active_groups:
            _, message_count = self.active_groups[group_id]
            # 消息越多，冷却时间越短
            return max(self.min_cooldown, self.max_cooldown - min(50, message_count) * 6)
        return self.max_cooldown
    
    async def should_activate(self, group_id: int) -> bool:
        """检查是否应该触发主动发言"""
        # 检查群组是否启用主动发言
        if not group_settings_manager.get_group_setting(group_id, "active_enabled", True):
            return False
            
        current_time = time.time()
        
        # 检查群组是否活跃
        if group_id not in self.active_groups:
            return False
            
        last_active, message_count = self.active_groups[group_id]
        
        # 检查消息数量是否足够
        if message_count < self.min_activity:
            return False
        
        # 检查冷却时间
        last_message = self.group_cooldowns.get(group_id, 0)
        cooldown = self.get_group_cooldown(group_id)
        if current_time - last_message < cooldown:
            return False
        
        # 概率触发
        if random.random() < self.active_speaker_probability:
            # 更新冷却时间
            self.group_cooldowns[group_id] = current_time
            return True
        
        return False
    
    async def generate_message(self, group_id: int) -> Tuple[str, Optional[int]]:
        """生成主动发言消息和要艾特的用户(如果有)"""
        # 获取活跃用户
        active_users = self.get_active_users(group_id)
        mention_user = None
        
        # 检查群组是否启用艾特功能
        at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
        
        # 概率决定是否艾特用户
        if at_enabled and active_users:
            # 80%概率艾特活跃用户，20%概率艾特不活跃用户以保持互动性
            if random.random() < 0.8 and random.random() < self.mention_probability:
                mention_user = random.choice(active_users)
            elif random.random() < self.include_inactive_users:
                # 获取所有用户，筛选出不活跃的
                all_users = self._get_all_users(group_id)
                inactive_users = [uid for uid in all_users if uid not in active_users]
                if inactive_users:
                    mention_user = random.choice(inactive_users)
        
        # 获取当前群聊主题和上下文
        current_topic = self._get_current_topic(group_id)
        context_messages = []
        
        if mention_user and mention_user in self.user_activity[group_id]:
            # 从用户上下文中获取消息
            context = list(enhanced_ai.context_memory.get((group_id, mention_user), deque()))
            if context:
                context_messages = context[-min(len(context), 3):]  # 取最后1-3条消息
        
        # 生成回复消息
        ai_reply = await self._generate_contextual_reply(group_id, mention_user, context_messages, current_topic)
        
        if not ai_reply:
            ai_reply = self._get_generic_reply(group_id) or "大家好呀~"
        
        # 如果消息以@开头，移除@以避免错误
        if ai_reply.startswith('@'):
            ai_reply = ai_reply.lstrip('@').strip()
        
        # 如果消息太长，截断
        if len(ai_reply) > 100:
            ai_reply = ai_reply[:97] + "..."
        
        # 添加表情增加亲和力
        if random.random() < self.emoji_probability:
            ai_reply = self._add_emoji(ai_reply)
        
        return ai_reply, mention_user
        
    def _get_all_users(self, group_id: int) -> List[int]:
        """获取群内所有用户(包括不活跃的)"""
        if group_id in self.user_last_active:
            return list(self.user_last_active[group_id].keys())
        return []
        
    def _get_current_topic(self, group_id: int) -> Optional[str]:
        """获取当前群聊主题"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                # 获取最近10条消息的关键词
                cursor.execute(
                    "SELECT keywords FROM chat_logs WHERE group_id = ? ORDER BY timestamp DESC LIMIT 10",
                    (group_id,)
                )
                recent_keywords = cursor.fetchall()
                
                if recent_keywords:
                    # 合并所有关键词
                    all_keywords = []
                    for keywords_json in recent_keywords:
                        if keywords_json[0]:
                            try:
                                keywords = json.loads(keywords_json[0])
                                all_keywords.extend(keywords)
                            except:
                                continue
                    
                    # 找出出现频率最高的关键词
                    if all_keywords:
                        from collections import Counter
                        keyword_counts = Counter(all_keywords)
                        most_common = keyword_counts.most_common(1)
                        if most_common and most_common[0][1] > 1:
                            return most_common[0][0]
        except Exception as e:
            logger.error(f"获取当前主题失败: {e}")
        return None
        
    async def _generate_contextual_reply(self, group_id: int, mention_user: Optional[int], 
                                       context_messages: List[str], topic: Optional[str]) -> str:
        """生成上下文相关的回复"""
        try:
            # 根据群聊氛围选择回复类型
            reply_type = self._choose_reply_type(group_id)
            
            if reply_type == "topic_related" and topic:
                # 生成与主题相关的回复
                return await self._generate_topic_reply(group_id, topic)
            elif reply_type == "context_related" and context_messages:
                # 基于上下文生成回复
                combined_context = " ".join(context_messages)
                if mention_user:
                    return await enhanced_ai.generate_response(group_id, mention_user, combined_context)
                else:
                    # 不针对特定用户的上下文回复
                    return await enhanced_ai.find_most_relevant(combined_context)
            elif reply_type == "greeting" and mention_user:
                # 向特定用户打招呼
                greetings = [f"@{mention_user} 最近怎么样呀？", 
                            f"@{mention_user} 刚刚在聊什么有趣的事情呀？", 
                            f"@{mention_user} 你刚才说的我很感兴趣呢~"]
                return random.choice(greetings)
            else:
                # 返回随机历史消息
                return await enhanced_ai.get_random_history()
        except Exception as e:
            logger.error(f"生成上下文回复失败: {e}")
            return None
        
    def _choose_reply_type(self, group_id: int) -> str:
        """选择回复类型"""
        # 根据群聊活跃度和设置调整权重
        weights = {
            "topic_related": 0.4,  # 与主题相关
            "context_related": 0.3,  # 与上下文相关
            "greeting": 0.2,  # 打招呼
            "random": 0.1  # 随机消息
        }
        
        # 根据群设置调整权重
        if not self.follow_topic:
            weights["topic_related"] = 0.2
            weights["random"] = 0.3
        
        # 根据权重随机选择
        types = list(weights.keys())
        probabilities = list(weights.values())
        return random.choices(types, probabilities, k=1)[0]
        
    async def _generate_topic_reply(self, group_id: int, topic: str) -> str:
        """生成与主题相关的回复"""
        try:
            # 获取与主题相关的消息
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT message FROM chat_logs WHERE (message LIKE ? OR keywords LIKE ?) AND quality_score > 0.7 ORDER BY RANDOM() LIMIT 5",
                    (f"%{topic}%", f"%{topic}%")
                )
                results = cursor.fetchall()
                
                if results:
                    replies = [row[0] for row in results]
                    # 选择一个回复并稍微修改
                    selected = random.choice(replies)
                    return self._adapt_reply_to_topic(selected, topic)
                
            # 如果没有找到相关消息，生成通用话题回复
            topic_replies = [
                f"最近大家好像经常讨论{topic}呢，你们怎么看？",
                f"关于{topic}，我觉得还挺有意思的~",
                f"说到{topic}，我之前看到过一些相关的内容..."
            ]
            return random.choice(topic_replies)
        except Exception as e:
            logger.error(f"生成主题回复失败: {e}")
            return None
        
    def _adapt_reply_to_topic(self, original_reply: str, topic: str) -> str:
        """调整回复以适应主题"""
        # 简单的适应性调整
        if topic in original_reply:
            return original_reply
        
        # 概率性地在回复中加入主题
        if random.random() < 0.7:
            adaptations = [
                f"{original_reply} 对了，说到{topic}...",
                f"关于{topic}，{original_reply}",
                f"{original_reply} 你们觉得{topic}怎么样？"
            ]
            return random.choice(adaptations)
        
        return original_reply
        
    def _get_generic_reply(self, group_id: int) -> str:
        """获取通用回复"""
        generic_replies = [
            "大家都在聊什么呀？",
            "今天群里好热闹呀~",
            "有人在吗？冒个泡呗~",
            "我也来凑个热闹~",
            "哈哈，这个话题很有意思呢！",
            "大家最近都在忙什么呀？",
            "有没有什么好玩的事情分享一下？"
        ]
        return random.choice(generic_replies)
        
    def _add_emoji(self, text: str) -> str:
        """为文本添加表情"""
        emojis = ["😊", "😂", "😉", "😆", "😘", "👍", "❤️", "✨", "🎉", "🤔"]
        
        # 随机选择表情位置
        position = random.choice(["start", "end", "both"])
        
        if position == "start":
            return f"{random.choice(emojis)} {text}"
        elif position == "end":
            return f"{text} {random.choice(emojis)}"
        else:
            return f"{random.choice(emojis)} {text} {random.choice(emojis)}"  # pyright: ignore[reportUnreachable, reportUnreachable]

active_speaker = ActiveSpeaker()

# ========== 相似度回复系统 ==========
class SimilarityReply:
    def __init__(self):
        # 回复冷却: {group_id: last_reply_time}
        self.reply_cooldowns: Dict[int, float] = {}
        # 最小相似度阈值
        self.similarity_threshold = 0.65
        # 回复概率
        self.reply_probability = 0.35
        # 最小冷却时间(秒)
        self.min_cooldown = 30
    
    async def should_reply(self, group_id: int, message: str) -> bool:
        """检查是否应该回复消息"""
        # 检查群组是否启用主动发言
        if not group_settings_manager.get_group_setting(group_id, "active_enabled", True):
            return False
            
        current_time = time.time()
        
        # 检查冷却时间
        last_reply = self.reply_cooldowns.get(group_id, 0)
        if current_time - last_reply < self.min_cooldown:
            return False
        
        # 获取最相关的历史消息
        relevant_reply = await enhanced_ai.find_most_relevant(message)
        if not relevant_reply:
            return False
        
        # 计算相似度
        if enhanced_ai.vectorizer:
            query_vec = enhanced_ai.vectorizer.transform([message])
            reply_vec = enhanced_ai.vectorizer.transform([relevant_reply])
            similarity = cosine_similarity(query_vec, reply_vec)[0][0]
        else:
            # 简单基于词重叠的相似度计算
            msg_words = set(jieba.lcut(message))
            reply_words = set(jieba.lcut(relevant_reply))
            similarity = len(msg_words & reply_words) / max(1, len(msg_words | reply_words))
        
        # 检查相似度和概率
        if similarity >= self.similarity_threshold and random.random() < self.reply_probability:
            self.reply_cooldowns[group_id] = current_time
            return True
        
        return False

similarity_reply = SimilarityReply()

# ========== 辅助插件主逻辑 ==========
assistant = on_message(priority=98, block=False)
@assistant.handle()
async def handle_assistant(bot: Bot, event: GroupMessageEvent):
    """处理主动发言和相似度回复"""
    try:
        group_id = event.group_id
        
        # 更新群组活跃度
        active_speaker.update_group_activity(group_id)
        
        # 更新用户活跃度
        active_speaker.update_user_activity(group_id, event.user_id)
        
        # 检查是否应该触发主动发言
        if await active_speaker.should_activate(group_id):
            # 生成主动发言消息
            message, mention_user = await active_speaker.generate_message(group_id)
            
            if mention_user:
                # 检查群组是否启用艾特功能
                at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
                if at_enabled:
                    # 艾特用户并发送消息
                    if config.at_reply_at_sender:
                        reply_msg = main_plugin.create_at_reply(mention_user, message)
                    else:
                        reply_msg = f"@{mention_user} {message}"
                    await bot.send_group_msg(group_id=group_id, message=reply_msg)
                    logger.info(f"主动发言(艾特): 群 {group_id} @{mention_user}: {message}")
                else:
                    # 群组禁用艾特，直接发送消息
                    await bot.send_group_msg(group_id=group_id, message=message)
                    logger.info(f"主动发言: 群 {group_id}: {message}")
            else:
                # 直接发送消息
                await bot.send_group_msg(group_id=group_id, message=message)
                logger.info(f"主动发言: 群 {group_id}: {message}")
        
        # 处理相似度回复
        raw_msg = event.get_plaintext().strip()
        if raw_msg and await similarity_reply.should_reply(group_id, raw_msg):
            # 获取相关回复
            ai_reply = await enhanced_ai.generate_response(group_id, event.user_id, raw_msg)
            if ai_reply:
                # 检查群组是否启用艾特功能
                at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
                # 随机决定是否艾特用户
                if at_enabled and config.at_reply_at_sender and random.random() < 0.5:
                    reply_msg = main_plugin.create_at_reply(event.user_id, ai_reply)
                else:
                    reply_msg = ai_reply
                
                await bot.send(event, reply_msg)
                logger.info(f"相似度回复: 群 {group_id}: {ai_reply[:30]}...")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"辅助插件处理消息时出错: {e}")

# ========== 主动发言管理命令 ==========
active_setting = on_command("主动发言设置", permission=SUPERUSER | GROUP_ADMIN | GROUP_OWNER, priority=1)
@active_setting.handle()
async def active_setting_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """设置主动发言参数"""
    try:
        group_id = event.group_id
        arg_str = args.extract_plain_text().strip()
        
        if not arg_str:
            # 显示当前设置
            active_enabled = group_settings_manager.get_group_setting(group_id, "active_enabled", True)
            at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
            
            current_settings = (
                f"群 {group_id} 主动发言设置:\n"
                f"启用状态: {'开启' if active_enabled else '关闭'}\n"
                f"艾特功能: {'开启' if at_enabled else '关闭'}\n"
                f"触发概率: {active_speaker.active_speaker_probability*100:.1f}%\n"
                f"最小冷却: {active_speaker.min_cooldown}秒\n"
                f"最大冷却: {active_speaker.max_cooldown}秒\n"
                f"最小活跃消息: {active_speaker.min_activity}条\n"
                f"艾特概率: {active_speaker.mention_probability*100:.1f}%\n"
                f"上下文窗口: {active_speaker.context_window}条"
            )
            await active_setting.finish(current_settings)
        
        # 解析参数
        params = {}
        for item in arg_str.split():
            if '=' in item:
                key, value = item.split('=', 1)
                params[key.strip()] = value.strip()
        
        # 更新设置
        updated = False
        
        if '启用' in params:
            enabled = params['启用'].lower() in ['true', '1', 'on', 'yes', '开启']
            group_settings_manager.set_group_setting(group_id, "active_enabled", enabled)
            updated = True
        
        if '艾特' in params:
            at_enabled = params['艾特'].lower() in ['true', '1', 'on', 'yes', '开启']
            group_settings_manager.set_group_setting(group_id, "at_enabled", at_enabled)
            updated = True
        
        if '概率' in params:
            new_prob = float(params['概率'])
            if 0 <= new_prob <= 1:
                active_speaker.active_speaker_probability = new_prob
                updated = True
            else:
                raise ValueError("概率必须在0-1之间")
        
        if '最小冷却' in params:
            min_cd = int(params['最小冷却'])
            if min_cd > 0:
                active_speaker.min_cooldown = min_cd
                updated = True
        
        if '最大冷却' in params:
            max_cd = int(params['最大冷却'])
            if max_cd > active_speaker.min_cooldown:
                active_speaker.max_cooldown = max_cd
                updated = True
        
        if '活跃消息' in params:
            min_act = int(params['活跃消息'])
            if min_act > 0:
                active_speaker.min_activity = min_act
                updated = True
        
        if '艾特概率' in params:
            mention_prob = float(params['艾特概率'])
            if 0 <= mention_prob <= 1:
                active_speaker.mention_probability = mention_prob
                updated = True
        
        if '上下文窗口' in params:
            context_win = int(params['上下文窗口'])
            if context_win > 0:
                active_speaker.context_window = context_win
                updated = True
        
        if updated:
            # 返回更新后的设置
            await active_setting_handler(bot, event, Message(""))
        else:
            await active_setting.finish("未识别到有效的设置参数")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"设置主动发言参数失败: {e}")
        await active_setting.finish(f"设置失败: {str(e)}")

test_active = on_command("测试主动发言", permission=SUPERUSER | GROUP_ADMIN | GROUP_OWNER, priority=1)
@test_active.handle()
async def test_active_handler(bot: Bot, event: GroupMessageEvent):
    """测试主动发言功能"""
    try:
        group_id = event.group_id
        
        # 检查群组是否启用主动发言
        if not group_settings_manager.get_group_setting(group_id, "active_enabled", True):
            await test_active.finish("本群已禁用主动发言功能")
        
        # 强制触发主动发言
        message, mention_user = await active_speaker.generate_message(group_id)
        
        # 检查群组是否启用艾特功能
        at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
        
        if mention_user and at_enabled:
            # 艾特用户并发送消息
            if config.at_reply_at_sender:
                reply_msg = main_plugin.create_at_reply(mention_user, message)
            else:
                reply_msg = f"@{mention_user} {message}"
            await bot.send(event, reply_msg)
            logger.info(f"测试主动发言(艾特): @{mention_user}: {message}")
        else:
            # 直接发送消息
            await bot.send(event, message)
            logger.info(f"测试主动发言: {message}")
        
        await test_active.finish("主动发言测试已触发")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"测试主动发言失败: {e}")
        await test_active.finish(f"测试失败: {str(e)}")

# ========== 群活跃度统计命令 ==========
group_activity = on_command("群活跃度", permission=SUPERUSER | GROUP_ADMIN | GROUP_OWNER, priority=1)
@group_activity.handle()
async def group_activity_handler(bot: Bot, event: GroupMessageEvent):
    """查看群活跃度统计"""
    try:
        group_id = event.group_id
        current_time = time.time()
        
        if group_id not in active_speaker.active_groups:
            await group_activity.finish("本群暂无活跃度数据")
        
        last_active, message_count = active_speaker.active_groups[group_id]
        minutes_ago = (current_time - last_active) / 60
        
        # 获取活跃用户数
        active_users = active_speaker.get_active_users(group_id)
        
        # 获取冷却时间信息
        cooldown = active_speaker.get_group_cooldown(group_id)
        last_cool = active_speaker.group_cooldowns.get(group_id, 0)
        cool_remain = max(0, cooldown - (current_time - last_cool))
        
        # 获取群设置
        active_enabled = group_settings_manager.get_group_setting(group_id, "active_enabled", True)
        at_enabled = group_settings_manager.get_group_setting(group_id, "at_enabled", True)
        
        msg = (
            f"群 {group_id} 活跃度统计:\n"
            f"主动发言: {'开启' if active_enabled else '关闭'}\n"
            f"艾特功能: {'开启' if at_enabled else '关闭'}\n"
            f"最近活跃: {minutes_ago:.1f}分钟前\n"
            f"消息数量: {message_count}条\n"
            f"活跃用户: {len(active_users)}人\n"
            f"冷却时间: {cool_remain:.0f}/{cooldown}秒\n"
            f"下次触发概率: {active_speaker.active_speaker_probability*100:.1f}%"
        )
        
        await group_activity.finish(msg)
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"获取群活跃度失败: {e}")
        await group_activity.finish("获取活跃度失败")

# ========== 相似度回复设置命令 ==========
similarity_setting = on_command("相似度设置", permission=SUPERUSER | GROUP_ADMIN | GROUP_OWNER, priority=1)
@similarity_setting.handle()
async def similarity_setting_handler(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """设置相似度回复参数"""
    try:
        arg_str = args.extract_plain_text().strip()
        
        if not arg_str:
            # 显示当前设置
            current_settings = (
                f"相似度回复设置:\n"
                f"阈值: {similarity_reply.similarity_threshold:.2f}\n"
                f"触发概率: {similarity_reply.reply_probability*100:.1f}%\n"
                f"冷却时间: {similarity_reply.min_cooldown}秒"
            )
            await similarity_setting.finish(current_settings)
        
        # 解析参数
        params = {}
        for item in arg_str.split():
            if '=' in item:
                key, value = item.split('=', 1)
                params[key.strip()] = value.strip()
        
        # 更新设置
        updated = False
        
        if '阈值' in params:
            new_threshold = float(params['阈值'])
            if 0.3 <= new_threshold <= 0.95:
                similarity_reply.similarity_threshold = new_threshold
                updated = True
            else:
                raise ValueError("阈值必须在0.3-0.95之间")
        
        if '概率' in params:
            new_prob = float(params['概率'])
            if 0 <= new_prob <= 1:
                similarity_reply.reply_probability = new_prob
                updated = True
        
        if '冷却' in params:
            new_cool = int(params['冷却'])
            if new_cool > 0:
                similarity_reply.min_cooldown = new_cool
                updated = True
        
        if updated:
            # 返回更新后的设置
            await similarity_setting_handler(bot, event, Message(""))
        else:
            await similarity_setting.finish("未识别到有效的设置参数")
    except FinishedException:
        pass
    except Exception as e:
        logger.error(f"设置相似度参数失败: {e}")
        await similarity_setting.finish(f"设置失败: {str(e)}")

# ========== 辅助插件初始化 ==========
def init_assistant():
    """初始化辅助插件"""
    logger.info("主动发言助手插件已初始化")

# 在插件加载时初始化
init_assistant()