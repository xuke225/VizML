# -*- coding: utf-8 -*-
"""
历史运行记录存储（SQLite）

按用户（user_id）保存每次训练的关键信息：
- 算法模块、数据集、超参数、评估指标、时间戳
仅记录可复现的参数与指标，不保存完整模型与可视化结果。
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'vizml_history.db',
)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT DEFAULT '',
            module TEXT DEFAULT '',
            label TEXT DEFAULT '',
            dataset TEXT DEFAULT '',
            params TEXT DEFAULT '{}',
            metrics TEXT DEFAULT '{}',
            replay TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    ''')
    # 兼容旧表：若无 replay 列则补上
    cols = [r['name'] for r in conn.execute('PRAGMA table_info(history)').fetchall()]
    if 'replay' not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN replay TEXT DEFAULT '{}'")
    conn.execute('CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id)')
    conn.commit()
    conn.close()


def _dump(value):
    return json.dumps(value or {}, ensure_ascii=False, default=str)


def save_record(user_id, module='', label='', dataset='', params=None, metrics=None, username='', replay=None):
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = _connect()
    cur = conn.execute(
        'INSERT INTO history '
        '(user_id, username, module, label, dataset, params, metrics, replay, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (user_id, username or '', module, label, dataset or '',
         _dump(params), _dump(metrics), _dump(replay), created_at),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id


def list_records(user_id, limit=200, offset=0):
    conn = _connect()
    rows = conn.execute(
        'SELECT * FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?',
        (user_id, limit, offset),
    ).fetchall()
    conn.close()
    return [{
        'id': r['id'],
        'user_id': r['user_id'],
        'username': r['username'],
        'module': r['module'],
        'label': r['label'],
        'dataset': r['dataset'],
        'params': json.loads(r['params'] or '{}'),
        'metrics': json.loads(r['metrics'] or '{}'),
        'replay': json.loads(r['replay'] or '{}'),
        'created_at': r['created_at'],
    } for r in rows]


def delete_record(record_id, user_id):
    conn = _connect()
    conn.execute('DELETE FROM history WHERE id = ? AND user_id = ?', (record_id, user_id))
    conn.commit()
    conn.close()


def clear_records(user_id):
    conn = _connect()
    conn.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()


def update_username(user_id, username):
    conn = _connect()
    conn.execute('UPDATE history SET username = ? WHERE user_id = ?', (username, user_id))
    conn.commit()
    conn.close()