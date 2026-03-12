"""Skills loader for agent capabilities."""
# 技能加载器模块 - 负责加载和管理 Agent 的技能
# 技能是 markdown 文件（SKILL.md），用于教 Agent 如何使用特定工具或执行特定任务

import json  # JSON 序列化
import os  # 操作系统接口
import re  # 正则表达式
import shutil  # 文件操作工具
from pathlib import Path  # 文件路径处理

# 默认内置技能目录（相对于本文件的位置）
# __file__ 是当前文件路径，parent.parent 是 nanobot/ 目录
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """
    # 技能加载器类 - 加载和管理 Agent 技能

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        # 构造函数，初始化技能加载器
        # workspace: 工作空间路径
        # builtin_skills_dir: 内置技能目录，None 则使用默认路径
        
        self.workspace = workspace  # 保存工作空间路径
        # 用户自定义技能目录
        self.user_skills_dir = workspace / "skills"
        # 内置技能目录（使用传入的或默认的）
        self.builtin_skills_dir = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def _list_skills(self) -> dict[str, Path]:
        """List all available skills from builtin and user directories."""
        # 列出所有可用技能，返回 技能名 -> 路径 的字典
        
        skills: dict[str, Path] = {}  # 技能字典
        
        # 先扫描内置技能（优先级较低，会被用户技能覆盖）
        if self.builtin_skills_dir.exists():  # 如果内置技能目录存在
            for skill_dir in self.builtin_skills_dir.iterdir():  # 遍历子目录
                if skill_dir.is_dir():  # 如果是目录
                    skill_file = skill_dir / "SKILL.md"  # 技能文件路径
                    if skill_file.exists():  # 如果技能文件存在
                        skills[skill_dir.name] = skill_file  # 添加到字典
        
        # 再扫描用户技能（优先级较高，会覆盖内置技能）
        if self.user_skills_dir.exists():  # 如果用户技能目录存在
            for skill_dir in self.user_skills_dir.iterdir():  # 遍历子目录
                if skill_dir.is_dir():  # 如果是目录
                    skill_file = skill_dir / "SKILL.md"  # 技能文件路径
                    if skill_file.exists():  # 如果技能文件存在
                        skills[skill_dir.name] = skill_file  # 添加到字典（覆盖内置）
        
        return skills

    def get_always_skills(self) -> list[str]:
        """Get list of skill names marked as always=true."""
        # 获取标记为 always=true 的技能列表
        # 这些技能会被自动加载到上下文中
        
        always = []  # 始终激活的技能列表
        for name, path in self._list_skills().items():  # 遍历所有技能
            content = path.read_text(encoding="utf-8")  # 读取技能文件内容
            # 检查是否包含 always=true 标记
            if re.search(r'^always:\s*true\s*$', content, re.MULTILINE | re.IGNORECASE):
                always.append(name)  # 添加到列表
        return always

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """Load skill content for injection into prompts."""
        # 加载技能内容，用于注入到提示词中
        # skill_names: 要加载的技能名称列表
        
        all_skills = self._list_skills()  # 获取所有技能
        parts = []  # 存储各部分内容的列表
        
        for name in skill_names:  # 遍历要加载的技能
            if name not in all_skills:  # 如果技能不存在
                continue  # 跳过
            
            skill_file = all_skills[name]  # 获取技能文件路径
            content = skill_file.read_text(encoding="utf-8")  # 读取内容
            
            # 移除 frontmatter（YAML 头部）
            # frontmatter 格式：---\nkey: value\n---
            if content.startswith("---"):
                parts_split = content.split("---", 2)
                if len(parts_split) >= 3:
                    content = parts_split[2].strip()  # 取第三部分（实际内容）
            
            # 添加技能标题和内容
            parts.append(f"## {name}\n\n{content}")
        
        # 用分隔线连接所有技能内容
        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self) -> str:
        """Build a summary table of all available skills."""
        # 构建所有可用技能的摘要表格
        
        skills = self._list_skills()  # 获取所有技能
        if not skills:  # 如果没有技能
            return ""
        
        lines = []  # 存储表格行
        lines.append("| Skill | Description | Available |")  # 表头
        lines.append("|-------|-------------|-----------|")  # 分隔线
        
        for name in sorted(skills.keys()):  # 按名称排序遍历
            path = skills[name]  # 获取技能路径
            content = path.read_text(encoding="utf-8")  # 读取内容
            
            # 解析 frontmatter
            description = ""  # 描述
            available = "✓"  # 默认可用
            
            if content.startswith("---"):
                # 提取 frontmatter
                frontmatter = content.split("---", 2)[1] if len(content.split("---")) > 2 else ""
                
                # 提取 description
                desc_match = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
                if desc_match:
                    description = desc_match.group(1).strip()
                
                # 提取 available
                avail_match = re.search(r'^available:\s*(\S+)', frontmatter, re.MULTILINE)
                if avail_match:
                    available = "✓" if avail_match.group(1).lower() == "true" else "✗"
            
            # 添加表格行
            lines.append(f"| {name} | {description} | {available} |")
        
        return "\n".join(lines)

    def get_skill_path(self, name: str) -> Path | None:
        """Get the path to a skill file, or None if not found."""
        # 获取技能文件路径，如果不存在则返回 None
        
        skills = self._list_skills()  # 获取所有技能
        return skills.get(name)  # 返回指定技能的路径

    def install_skill(self, name: str, source: Path) -> bool:
        """Install a skill from source directory to user skills."""
        # 从源目录安装技能到用户技能目录
        # name: 技能名称
        # source: 源目录路径
        # 返回是否成功
        
        if not source.exists() or not source.is_dir():  # 检查源目录
            return False
        
        skill_file = source / "SKILL.md"  # 技能文件路径
        if not skill_file.exists():  # 检查技能文件
            return False
        
        # 目标路径
        target_dir = self.user_skills_dir / name
        
        try:
            # 如果目标已存在，先删除
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            # 复制目录
            shutil.copytree(source, target_dir)
            return True
        except Exception:
            return False

    def uninstall_skill(self, name: str) -> bool:
        """Uninstall a user skill."""
        # 卸载用户技能
        # 只能卸载用户技能，不能卸载内置技能
        
        target_dir = self.user_skills_dir / name
        if not target_dir.exists():  # 如果不存在
            return False
        
        try:
            shutil.rmtree(target_dir)  # 删除目录
            return True
        except Exception:
            return False
