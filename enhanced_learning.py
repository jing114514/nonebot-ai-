import asyncio
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import jieba
from collections import defaultdict
import logging

# 设置日志
logger = logging.getLogger('Ben_learning_chat.enhanced_learning')

class EnhancedTrainingSystem:
    """增强的训练系统，负责自主训练和效果评估"""
    def __init__(self, db_path: str, config):
        self.db_path = db_path
        self.config = config
        
        # 自主训练相关配置
        self.last_training_time = 0
        self.message_count_since_last_training = 0
        self.training_scheduler = None
        
        # 效果评估指标
        self.performance_metrics = {
            'reply_relevance': [],  # 回复相关性评分
            'conversation_coherence': [],  # 对话连贯性评分
            'user_satisfaction': [],  # 用户满意度（基于回复后的互动）
            'training_efficiency': [],  # 训练效率（时间/质量）
        }
        
        # 高级模型存储
        self.vectorizer = None
        self.word_embeddings = None
        self.cluster_model = None
        
        # 初始化评估数据库
        self._init_metrics_db()
    
    def _init_metrics_db(self):
        """初始化性能指标数据库"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 创建性能指标表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    metric_type TEXT,
                    value REAL,
                    details TEXT
                )
                ''')
                # 创建训练历史表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    message_count INTEGER,
                    duration REAL,
                    model_version TEXT,
                    status TEXT,
                    metrics TEXT
                )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"初始化指标数据库失败: {e}")
    
    async def start_auto_training(self):
        """启动自动训练调度器"""
        if self.training_scheduler and not self.training_scheduler.done():
            logger.warning("自动训练调度器已经在运行")
            return
            
        logger.info("启动自动训练调度器")
        self.training_scheduler = asyncio.create_task(self._auto_training_loop())
    
    async def stop_auto_training(self):
        """停止自动训练调度器"""
        if self.training_scheduler and not self.training_scheduler.done():
            self.training_scheduler.cancel()
            try:
                await self.training_scheduler
            except asyncio.CancelledError:
                logger.info("自动训练调度器已停止")
            except Exception as e:
                logger.error(f"停止训练调度器时出错: {e}")
        else:
            logger.info("自动训练调度器未运行")
    
    async def _auto_training_loop(self):
        """自动训练主循环"""
        while True:
            try:
                # 检查是否需要训练
                if await self._should_train():
                    logger.info("触发自动训练")
                    await self.perform_training()
                    
                # 等待下次检查
                await asyncio.sleep(self.config.auto_training_check_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"自动训练循环出错: {e}")
                # 发生错误后等待更长时间再尝试
                await asyncio.sleep(max(60, self.config.auto_training_check_interval))
    
    async def _should_train(self) -> bool:
        """判断是否应该进行训练"""
        current_time = time.time()
        
        # 检查时间间隔
        time_condition = current_time - self.last_training_time >= self.config.auto_training_interval
        
        # 检查消息数量
        message_condition = self.message_count_since_last_training >= self.config.auto_training_min_messages
        
        return time_condition or message_condition
    
    async def perform_training(self):
        """执行训练过程"""
        import sqlite3
        
        start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"开始训练AI模型，时间: {start_time_str}")
        
        try:
            # 获取训练数据
            messages = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT message FROM chat_logs WHERE quality_score >= ? ORDER BY RANDOM() LIMIT ?",
                    (self.config.message_quality_threshold, self.config.training_max_messages)
                )
                messages = [row[0] for row in cursor.fetchall()]
            
            if not messages:
                logger.warning("没有足够的高质量消息用于训练")
                return
            
            message_count = len(messages)
            logger.info(f"使用 {message_count} 条消息进行训练")
            
            # 执行高级训练
            await self._advanced_training(messages)
            
            # 记录训练完成时间
            duration = time.time() - start_time
            self.last_training_time = time.time()
            self.message_count_since_last_training = 0
            
            # 保存训练历史
            metrics_summary = self._get_current_metrics_summary()
            self._save_training_history(message_count, duration, metrics_summary)
            
            logger.info(f"AI模型训练完成，耗时: {duration:.2f}秒")
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            # 记录失败状态
            duration = time.time() - start_time
            self._save_training_history(0, duration, {}, status="failed", error=str(e))
    
    async def _advanced_training(self, messages: List[str]):
        """执行高级训练算法"""
        try:
            # 1. 训练TF-IDF向量器（增强版）
            self._train_enhanced_tfidf(messages)
            
            # 2. 训练K-means聚类模型
            self._train_clustering_model(messages)
            
            # 3. 保存训练结果
            self._save_model_state()
        except Exception as e:
            logger.error(f"高级训练失败: {e}")
            raise
    
    def _train_enhanced_tfidf(self, messages: List[str]):
        """训练增强版TF-IDF向量器"""
        # 自定义分词函数
        def tokenize(text):
            return list(jieba.cut_for_search(text))
        
        # 配置增强的TF-IDF向量器
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            max_features=self.config.tfidf_max_features,
            min_df=self.config.tfidf_min_df,
            max_df=self.config.tfidf_max_df,
            ngram_range=(1, self.config.tfidf_ngram_range),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # 训练向量器
        self.vectorizer.fit(messages)
        logger.info(f"TF-IDF向量器训练完成，词汇表大小: {len(self.vectorizer.vocabulary_)}")
    
    def _train_clustering_model(self, messages: List[str]):
        """训练对话聚类模型"""
        if not self.vectorizer or len(messages) < self.config.cluster_min_samples:
            logger.warning("跳过聚类模型训练，条件不满足")
            return
        
        # 将消息转换为向量
        message_vectors = self.vectorizer.transform(messages)
        
        # 训练K-means模型
        n_clusters = min(self.config.cluster_max_clusters, max(2, len(messages) // 100))
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        self.cluster_model.fit(message_vectors)
        
        logger.info(f"聚类模型训练完成，聚类数量: {n_clusters}")
    
    def _save_model_state(self):
        """保存模型状态"""
        # 实际项目中可能需要使用pickle或joblib保存模型
        # 这里简化处理，记录模型状态
        logger.info("模型状态已更新")
    
    def _simple_keyword_match(self, message: str, candidates: List[str]) -> Optional[str]:
        """当向量匹配失败时，使用简单的关键词匹配作为后备方案"""
        try:
            # 提取消息关键词
            message_keywords = set(jieba.cut_for_search(message))
            
            if not message_keywords:
                return None
            
            best_match = None
            best_score = 0
            
            # 对每个候选回复计算关键词匹配度
            for candidate in candidates:
                candidate_keywords = set(jieba.cut_for_search(candidate))
                intersection = message_keywords.intersection(candidate_keywords)
                
                # 计算匹配得分
                if len(candidate_keywords) > 0:
                    match_score = len(intersection) / len(candidate_keywords)
                else:
                    match_score = 0
                
                # 长度匹配奖励
                message_len = len(message)
                candidate_len = len(candidate)
                length_ratio = candidate_len / max(message_len, 1)
                
                # 如果长度差异太大，降低分数
                if length_ratio < 0.3 or length_ratio > 3.0:
                    match_score *= 0.5
                
                # 更新最佳匹配
                if match_score > best_score:
                    best_score = match_score
                    best_match = candidate
            
            # 如果匹配度足够高，返回最佳匹配
            if best_match and best_score > 0.2:  # 0.2是一个较低的阈值，确保至少有一定匹配度
                return best_match
            
            return None
        except Exception as e:
            logger.error(f"简单关键词匹配失败: {e}")
            return None
    
    def _save_training_history(self, message_count: int, duration: float, metrics: Dict, status: str = "success", error: str = ""):
        """保存训练历史记录"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO training_history 
                    (timestamp, message_count, duration, model_version, status, metrics) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        int(time.time()),
                        message_count,
                        duration,
                        self.config.model_version,
                        status,
                        json.dumps({**metrics, "error": error})
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
    
    def record_metric(self, metric_type: str, value: float, details: Dict = None):
        """记录性能指标"""
        import sqlite3
        
        # 更新内存中的指标
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
            # 保持指标列表不要太长
            if len(self.performance_metrics[metric_type]) > self.config.metric_history_size:
                self.performance_metrics[metric_type] = self.performance_metrics[metric_type][-self.config.metric_history_size:]
        
        # 保存到数据库
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO performance_metrics 
                    (timestamp, metric_type, value, details) 
                    VALUES (?, ?, ?, ?)""",
                    (
                        int(time.time()),
                        metric_type,
                        value,
                        json.dumps(details) if details else ""
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"保存指标失败: {e}")
    
    def _get_current_metrics_summary(self) -> Dict:
        """获取当前指标摘要"""
        summary = {}
        for metric_type, values in self.performance_metrics.items():
            if values:
                summary[f"{metric_type}_avg"] = np.mean(values)
                summary[f"{metric_type}_max"] = np.max(values)
                summary[f"{metric_type}_min"] = np.min(values)
                summary[f"{metric_type}_count"] = len(values)
        return summary
    
    async def evaluate_reply_quality(self, query: str, reply: str) -> float:
        """评估回复质量"""
        if not query or not reply:
            return 0.0
        
        try:
            # 基础相关性评分
            relevance_score = self._calculate_relevance_score(query, reply)
            
            # 长度适宜性评分
            length_score = self._calculate_length_score(query, reply)
            
            # 内容质量评分
            content_score = self._calculate_content_quality_score(reply)
            
            # 综合评分
            final_score = (relevance_score * 0.5 + length_score * 0.2 + content_score * 0.3)
            
            # 记录指标
            self.record_metric('reply_relevance', final_score, {
                'query': query[:50],
                'reply': reply[:50]
            })
            
            return final_score
        except Exception as e:
            logger.error(f"评估回复质量失败: {e}")
            return 0.0
    
    def _calculate_relevance_score(self, query: str, reply: str) -> float:
        """计算查询与回复的相关性评分"""
        if not self.vectorizer:
            # 简化的相关性计算
            query_words = set(jieba.cut_for_search(query))
            reply_words = set(jieba.cut_for_search(reply))
            intersection = query_words.intersection(reply_words)
            
            if not query_words:
                return 0.5  # 默认值
            
            return len(intersection) / len(query_words)
        
        try:
            # 使用TF-IDF和余弦相似度计算相关性
            vectors = self.vectorizer.transform([query, reply])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except Exception as e:
            logger.error(f"计算相关性失败: {e}")
            # 回退到简单方法
            query_words = set(jieba.cut_for_search(query))
            reply_words = set(jieba.cut_for_search(reply))
            intersection = query_words.intersection(reply_words)
            
            if not query_words:
                return 0.5  # 默认值
            
            return len(intersection) / len(query_words)
    
    def _calculate_length_score(self, query: str, reply: str) -> float:
        """计算回复长度适宜性评分"""
        query_length = len(query)
        reply_length = len(reply)
        
        if query_length == 0:
            return 0.5  # 默认值
        
        # 理想的回复长度应该在查询长度的0.5到2倍之间
        if 0.5 * query_length <= reply_length <= 2.0 * query_length:
            return 1.0
        elif reply_length < 0.5 * query_length:
            return reply_length / (0.5 * query_length) * 0.5 + 0.5
        else:
            # 过长的回复，评分递减
            ratio = 2.0 * query_length / reply_length
            return max(0.3, ratio)
    
    def _calculate_content_quality_score(self, reply: str) -> float:
        """计算回复内容质量评分"""
        # 检查是否为空
        if not reply.strip():
            return 0.0
        
        # 检查是否包含禁用词
        for word in self.config.prohibited_words:
            if word in reply:
                return 0.0
        
        # 检查句子完整性（简单检查是否包含句号、问号、感叹号等）
        has_ending_punctuation = any(p in reply for p in ["。", "!", "！", "?", "？", "."])
        
        # 计算词汇多样性
        words = jieba.cut_for_search(reply)
        unique_words = set(words)
        word_count = len(list(jieba.cut_for_search(reply)))  # 重新分词计算数量
        
        if word_count == 0:
            diversity_score = 0.0
        else:
            diversity_score = min(1.0, len(unique_words) / word_count)
        
        # 综合评分
        punctuation_score = 0.8 if has_ending_punctuation else 0.5
        
        return (diversity_score * 0.6 + punctuation_score * 0.4)
    
    async def get_performance_report(self, days: int = 7) -> Dict:
        """获取性能报告"""
        import sqlite3
        
        report = {
            'time_range': f"最近{days}天",
            'training_sessions': [],
            'metrics_summary': {},
            'improvement_trends': {}
        }
        
        try:
            cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取训练会话
                cursor.execute(
                    "SELECT timestamp, message_count, duration, status, metrics FROM training_history WHERE timestamp > ? ORDER BY timestamp DESC",
                    (cutoff_time,)
                )
                training_sessions = cursor.fetchall()
                
                for ts, msg_count, duration, status, metrics_json in training_sessions:
                    try:
                        metrics = json.loads(metrics_json) if metrics_json else {}
                    except json.JSONDecodeError:
                        metrics = {}
                    
                    report['training_sessions'].append({
                        'time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                        'message_count': msg_count,
                        'duration': duration,
                        'status': status,
                        'metrics': metrics
                    })
                
                # 获取性能指标
                for metric_type in self.performance_metrics.keys():
                    cursor.execute(
                        "SELECT AVG(value), MAX(value), MIN(value), COUNT(*) FROM performance_metrics WHERE metric_type = ? AND timestamp > ?",
                        (metric_type, cutoff_time)
                    )
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        report['metrics_summary'][metric_type] = {
                            'average': float(result[0]),
                            'max': float(result[1]),
                            'min': float(result[2]),
                            'count': int(result[3])
                        }
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
        
        return report
    
    def increment_message_count(self):
        """增加消息计数"""
        self.message_count_since_last_training += 1
        
        # 如果消息计数达到阈值，触发即时训练检查
        if self.message_count_since_last_training % self.config.message_count_check_interval == 0:
            asyncio.create_task(self._check_and_train_if_needed())
    
    async def _check_and_train_if_needed(self):
        """检查并在需要时触发训练"""
        if await self._should_train():
            await self.perform_training()
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'model_version': self.config.model_version,
            'last_training_time': datetime.fromtimestamp(self.last_training_time).strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else '从未训练',
            'message_count_since_last_training': self.message_count_since_last_training,
            'is_trained': self.vectorizer is not None,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        }
        
        return info
    
    def find_relevant_reply(self, message: str) -> Optional[str]:
        """根据输入消息找到最相关的回复"""
        if not message or not self.vectorizer:
            return None
        
        try:
            # 检查vectorizer是否已经训练过
            if not hasattr(self.vectorizer, 'vocabulary_') or len(self.vectorizer.vocabulary_) == 0:
                logger.warning("向量器尚未训练，无法进行相似度匹配")
                return None
            
            # 从数据库获取高质量消息用于匹配
            import sqlite3
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 获取部分高质量消息进行匹配
                cursor.execute(
                    "SELECT message FROM chat_logs WHERE quality_score >= ? ORDER BY RANDOM() LIMIT ?",
                    (self.config.message_quality_threshold, self.config.reply_candidate_limit)
                )
                candidates = [row[0] for row in cursor.fetchall()]
            
            if not candidates:
                return None
            
            # 使用增强的TF-IDF向量器进行相似度计算
            all_texts = [message] + candidates
            
            # 尝试转换文本为向量，捕获可能的错误
            try:
                vectors = self.vectorizer.transform(all_texts)
            except ValueError as e:
                logger.error(f"向量转换失败: {e}")
                # 如果转换失败，尝试使用简单的关键词匹配
                return self._simple_keyword_match(message, candidates)
            
            # 计算查询与所有候选消息的相似度
            similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
            
            # 找到相似度最高的消息
            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            
            # 如果相似度低于阈值，不返回任何内容
            if best_similarity < self.config.min_similarity:
                return None
            
            best_reply = candidates[best_index]
            
            # 记录性能指标
            self.record_metric('reply_relevance', best_similarity, {
                'query': message[:50],
                'reply': best_reply[:50],
                'similarity': best_similarity
            })
            
            return best_reply
        except Exception as e:
            logger.error(f"查找相关回复失败: {e}")
            return None
    
    def add_training_data(self, message: str, quality_score: float):
        """添加新的训练数据"""
        if not message:
            return
        
        try:
            # 增加消息计数，可能触发训练
            self.increment_message_count()
            
            # 这里可以添加更多的预处理和数据增强逻辑
            # 例如，可以根据quality_score决定是否立即加入训练集
            if quality_score > self.config.high_quality_threshold:
                # 高质量消息可以特殊标记或优先用于训练
                pass
        except Exception as e:
            logger.error(f"添加训练数据失败: {e}")

# 高级语义分析器
class AdvancedSemanticAnalyzer:
    """高级语义分析器，提供更复杂的文本分析功能"""
    def __init__(self):
        # 初始化情感词典和其他资源
        self._init_resources()
    
    def _init_resources(self):
        """初始化分析资源"""
        # 情感词汇表（简化版）
        self.positive_words = {
            '好': 1.0, '优秀': 2.0, '棒': 1.5, '开心': 1.2, '高兴': 1.2,
            '喜欢': 1.0, '满意': 1.5, '成功': 1.8, '精彩': 1.5, '赞美': 1.2,
            '不错': 0.8, '真好': 1.5, '太棒了': 2.0, '完美': 2.0
        }
        
        self.negative_words = {
            '不好': -1.0, '糟糕': -2.0, '差': -1.5, '失望': -1.2, '难过': -1.2,
            '讨厌': -1.0, '不满意': -1.5, '失败': -1.8, '无聊': -1.5, '批评': -1.2,
            '错误': -1.0, '坏': -1.5, '太差了': -2.0, '糟糕透了': -2.0
        }
        
        # 程度副词
        self.intensifiers = {
            '很': 1.5, '非常': 2.0, '特别': 1.8, '极其': 2.5, '太': 1.8,
            '有点': 0.5, '稍微': 0.3, '略微': 0.2
        }
        
        # 否定词
        self.negations = {'不', '没', '无', '非', '未', '否'}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'score': 0.0}
        
        # 分词
        words = list(jieba.cut(text))
        
        positive_score = 0.0
        negative_score = 0.0
        
        # 情感分析逻辑
        i = 0
        while i < len(words):
            word = words[i]
            
            # 检查是否是否定词
            is_negated = False
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            
            # 检查是否是程度副词
            intensifier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier = self.intensifiers[words[i-1]]
            
            # 检查情感词
            if word in self.positive_words:
                score = self.positive_words[word] * intensifier
                if is_negated:
                    negative_score += score
                else:
                    positive_score += score
            elif word in self.negative_words:
                score = abs(self.negative_words[word]) * intensifier
                if is_negated:
                    positive_score += score
                else:
                    negative_score += score
            
            i += 1
        
        # 计算综合得分
        overall_score = (positive_score - negative_score) / max(1.0, len(words) / 10.0)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'score': overall_score
        }
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """提取关键短语"""
        if not text:
            return []
        
        # 分词并过滤停用词
        words = list(jieba.cut_for_search(text))
        
        # 简单的词频统计
        word_freq = defaultdict(int)
        for word in words:
            # 过滤短词和标点
            if len(word) > 1 and word.strip() and not word.isspace():
                word_freq[word] += 1
        
        # 按词频排序
        sorted_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前N个短语
        return [{'phrase': phrase, 'frequency': freq} for phrase, freq in sorted_phrases[:top_n]]
    
    def detect_question_type(self, text: str) -> str:
        """检测问题类型"""
        if not text:
            return 'unknown'
        
        # 检查疑问词
        question_words = {
            'what': {'什么', '啥', '何', '何等', '哪些', '哪类'},
            'who': {'谁', '哪个', '哪些人', '哪位', '何人'},
            'when': {'何时', '什么时候', '何时', '哪天', '哪年', '几点', '何时'},
            'where': {'哪里', '何处', '何地', '哪儿', '什么地方'},
            'why': {'为什么', '为何', '何故', '原因', '为啥'},
            'how': {'如何', '怎么', '怎样', '怎么样', '如何做', '怎么做', '如何办'},
            'yes_no': {'吗', '么', '是否', '是不是', '对吗', '对么'}
        }
        
        # 检查结尾标点
        if text.endswith('?') or text.endswith('？'):
            for q_type, words in question_words.items():
                for word in words:
                    if word in text:
                        return q_type
            return 'open_ended'
        
        # 检查是否包含疑问词但没有问号
        for q_type, words in question_words.items():
            for word in words:
                if word in text:
                    return q_type
        
        return 'statement'
    
    def calculate_readability(self, text: str) -> float:
        """计算文本可读性（简化版）"""
        if not text:
            return 0.0
        
        # 统计字数
        char_count = len(text)
        if char_count == 0:
            return 0.0
        
        # 统计句子数（简单按句号、问号、感叹号分割）
        sentences = [s for s in re.split(r'[。！？!?]', text) if s.strip()]
        sentence_count = max(1, len(sentences))
        
        # 分词并统计词数
        words = list(jieba.cut(text))
        word_count = len(words)
        
        # 计算平均句子长度（字数）
        avg_sentence_length = char_count / sentence_count
        
        # 计算平均词长
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # 计算可读性得分（简化模型）
        # 得分越高表示越容易理解
        readability_score = 100 - (avg_sentence_length * 0.5 + (10 - avg_word_length) * 5)
        
        # 限制在0-100范围内
        return max(0.0, min(100.0, readability_score))

# 导入必要的模块
import re