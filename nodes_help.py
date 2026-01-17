"""
FXTD Studio Radiance - Help Node
Opens documentation in the user's default browser.
"""

import os
import webbrowser


class FXTDHelp:
    """
    Opens FXTD Studio Radiance documentation in your browser.
    """
    
    DOCS_OPTIONS = [
        "Installation Guide",
        "Full Documentation",
        "Node Reference",
        "Tutorials",
        "Workflows",
        "FAQ & Troubleshooting",
        "Changelog",
        "Pro Viewer Guide",
        "FXTD Website",
        "GitHub Repository",
    ]
    
    def __init__(self):
        self.docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "documentation": (cls.DOCS_OPTIONS, {"default": "Full Documentation"}),
            },
            "optional": {
                "trigger": ("*", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "open_docs"
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = "Open FXTD Studio Radiance documentation or website in your browser."
    OUTPUT_NODE = True
    
    def open_docs(self, documentation: str, trigger=None):
        """Open the selected documentation."""
        
        docs_map = {
            "Installation Guide": os.path.join(self.docs_dir, "installation.html"),
            "Full Documentation": os.path.join(self.docs_dir, "index.html"),
            "Node Reference": os.path.join(self.docs_dir, "node_reference.html"),
            "Tutorials": os.path.join(self.docs_dir, "tutorials.html"),
            "Workflows": os.path.join(self.docs_dir, "workflows.html"),
            "FAQ & Troubleshooting": os.path.join(self.docs_dir, "faq.html"),
            "Changelog": os.path.join(self.docs_dir, "changelog.html"),
            "Pro Viewer Guide": os.path.join(self.docs_dir, "radiance_viewer.html"),
            "FXTD Website": "https://www.fxtd.org",
            "GitHub Repository": "https://github.com/FXTDStudio/Radiance",
        }
        
        target = docs_map.get(documentation, docs_map["Full Documentation"])
        
        try:
            if target.startswith("http"):
                webbrowser.open(target)
                return (f"✅ Opened {documentation} in browser",)
            else:
                # Local file - convert to file:// URL
                if os.path.exists(target):
                    file_url = f"file:///{target.replace(os.sep, '/')}"
                    webbrowser.open(file_url)
                    return (f"✅ Opened {documentation} in browser",)
                else:
                    return (f"❌ Documentation not found: {target}",)
        except Exception as e:
            return (f"❌ Error opening docs: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "FXTDHelp": FXTDHelp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDHelp": "❓ Radiance Help & Docs",
}
