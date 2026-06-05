#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus数据库备份工具
支持将一个数据库的所有集合和数据备份到另一个数据库
V5 - 在恢复/备份后自动加载集合
"""

import json
import os
from typing import Dict, List, Any
from pymilvus import connections, Collection, utility, db
import argparse


class MilvusBackup:
    """Milvus数据库备份工具"""
    
    def __init__(self, config_path: str = "config/kg_config.json"):
        """初始化备份工具"""
        self.config = self._load_config(config_path)
        self.milvus_config = self.config.get("milvus", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {
                    "host": "localhost", 
                    "port": 19530
                }
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def connect_to_database(self, db_name: str) -> None:
        """连接到指定数据库"""
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        
        try:
            connections.connect(alias="default", host=host, port=port)
        except Exception:
            pass
        
        if db_name not in db.list_database():
            print(f"📝 创建数据库: {db_name}")
            db.create_database(db_name)
        
        connections.disconnect("default")
        connections.connect(alias="default", host=host, port=port, db_name=db_name)
        print(f"✅ 已连接到数据库: {db_name}")
    
    def list_collections(self) -> List[str]:
        """列出当前数据库中的所有集合"""
        return utility.list_collections()
    
    def backup_collection(self, collection_name: str, source_db: str, target_db: str) -> bool:
        """备份单个集合"""
        try:
            print(f"📦 备份集合: {collection_name}")
            
            self.connect_to_database(source_db)
            source_collection = Collection(collection_name)
            if not source_collection.has_index():
                print(f"  ⚠️ 源集合 {collection_name} 没有索引，将直接查询。")
            source_collection.load()
            
            schema = source_collection.schema
            index_info = source_collection.indexes
            
            primary_key_field = next((f.name for f in schema.fields if f.is_primary), None)
            if not primary_key_field:
                raise ValueError(f"集合 {collection_name} 中未找到主键字段。")

            query_expr = f'{primary_key_field} != ""'
            all_results = source_collection.query(expr=query_expr, output_fields=["*"], limit=16384)
            
            print(f"  📊 源集合记录数: {len(all_results)}")
            
            self.connect_to_database(target_db)
            
            if not utility.has_collection(collection_name):
                target_collection = Collection(collection_name, schema)
                print(f"  ✅ 创建目标集合: {collection_name}")
            else:
                target_collection = Collection(collection_name)
                print(f"  ✅ 目标集合已存在: {collection_name}")
            
            if all_results:
                target_collection.insert(all_results)
                target_collection.flush()
                print(f"  📝 插入数据: {len(all_results)} 条记录")
            
            for index in index_info:
                if not target_collection.has_index(index_name=index.index_name):
                    try:
                        target_collection.create_index(
                            field_name=index.field_name,
                            index_params=index.params,
                            index_name=index.index_name
                        )
                        print(f"  🔍 创建索引 '{index.index_name}' 在字段 '{index.field_name}' 上")
                    except Exception as e:
                        print(f"  ⚠️ 创建索引失败: {e}。可能是索引已存在。")

            # <<< MODIFIED: Add load operation after creating index
            print(f"  ⏳ 正在加载集合 {collection_name} 到内存...")
            target_collection.load()
            print(f"  ✅ 集合 {collection_name} 已加载。")

            return True
            
        except Exception as e:
            print(f"❌ 备份集合 {collection_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def backup_database(self, source_db: str, target_db: str, overwrite: bool = False) -> None:
        """备份整个数据库"""
        print(f"🚀 开始备份数据库: {source_db} -> {target_db}" + ("（覆盖模式）" if overwrite else ""))
        
        if overwrite:
            self.clear_database(target_db)
        
        self.connect_to_database(source_db)
        collections = self.list_collections()
        
        if not collections:
            print("ℹ️ 源数据库中没有集合可备份。")
            print("🎉 备份完成。")
            return
            
        print(f"📋 发现 {len(collections)} 个集合: {collections}")
        
        success_count = 0
        for collection_name in collections:
            if self.backup_collection(collection_name, source_db, target_db):
                success_count += 1
        
        print(f"🎉 备份完成: {success_count}/{len(collections)} 个集合备份成功")

    def clear_database(self, db_name: str) -> None:
        """删除指定数据库中的所有集合，删除前先释放"""
        print(f"🗑️ 开始清空数据库: {db_name}")
        self.connect_to_database(db_name)
        collections = self.list_collections()
        
        if not collections:
            print("ℹ️ 数据库中没有集合可删除。")
            return
            
        dropped_count = 0
        for collection_name in collections:
            try:
                try:
                    collection = Collection(collection_name)
                    if utility.has_collection(collection_name) and collection.has_index():
                        print(f"  - 正在释放集合: {collection_name}")
                        collection.release()
                except Exception as release_e:
                    print(f"  - 集合 {collection_name} 无需释放或释放失败: {str(release_e)[:100]}...")

                utility.drop_collection(collection_name)
                dropped_count += 1
                print(f"  🗑️ 已删除集合: {collection_name}")
            except Exception as e:
                print(f"  ⚠️ 删除集合 {collection_name} 失败: {e}")
        
        print(f"✅ 清空完成: 已删除 {dropped_count}/{len(collections)} 个集合")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus数据库备份工具")
    parser.add_argument("--source", type=str, help="源数据库名称")
    parser.add_argument("--target", type=str, help="目标数据库名称")
    parser.add_argument("--config", type=str, default="config/kg_config.json", help="配置文件路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖模式：清空目标库后再备份")
    
    args = parser.parse_args()
    
    if args.source and args.target:
        backup_tool = MilvusBackup(args.config)
        backup_tool.backup_database(args.source, args.target, overwrite=args.overwrite)
        return

    print("🔧 Milvus数据库备份工具")
    print("=" * 50)
    print("1. code_op -> code_op1 (覆盖备份)")
    print("2. code_op1 -> code_op (覆盖备份)")
    print("3. 清空 code_op 数据库")
    print("4. code_op -> code_op2 (覆盖备份)")
    print("5. code_op -> code_op_test (覆盖备份)")
    print("6. code_op_test -> code_op (覆盖恢复)")
    print("0. 退出")
    
    choice = input("请选择操作 (0-6): ").strip()
    
    backup_tool = MilvusBackup()
    
    if choice == "1":
        backup_tool.backup_database("code_op", "code_op1", overwrite=True)
    elif choice == "2":
        backup_tool.backup_database("code_op1", "code_op", overwrite=True)
    elif choice == "3":
        backup_tool.clear_database("code_op")
    elif choice == "4":
        backup_tool.backup_database("code_op", "code_op2", overwrite=True)
    elif choice == "5":
        backup_tool.backup_database("code_op", "code_op_test", overwrite=True)
    elif choice == "6":
        backup_tool.backup_database("code_op_test", "code_op", overwrite=True)
    elif choice == "0":
        print("👋 再见！")
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()