import os
import sys
import importlib.util
import re

# Add parent to path for sub-module imports (like PyOpenColorIO if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Robust Mock
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return True

sys.modules["folder_paths"] = MockModule()
sys.modules["folder_paths"].get_temp_directory = lambda: "temp"
sys.modules["comfy"] = MockModule()
sys.modules["comfy.utils"] = MockModule()
sys.modules["comfy.samplers"] = MockModule()
sys.modules["comfy.sample"] = MockModule()
sys.modules["comfy.model_management"] = MockModule()

# Mock Optional Dependencies
for mod in ["OpenEXR", "Imath", "cv2", "colour", "scipy", "scipy.special", "skimage", "skimage.filters", "torch", "numpy", "transformers", "opencolorio", "PyOpenColorIO"]:
    sys.modules[mod] = MockModule()

sys.modules["comfy.sd"] = MockModule()
sys.modules["torch.nn"] = MockModule()
sys.modules["torch.nn.functional"] = MockModule()
sys.modules["PIL"] = MockModule()
sys.modules["PIL.Image"] = MockModule()
sys.modules["PIL.ImageOps"] = MockModule()
sys.modules["PIL.ImageFilter"] = MockModule()
sys.modules["PIL.ImageDraw"] = MockModule()
sys.modules["PIL.ImageFont"] = MockModule()

def load_nodes_from_file(filepath):
    filename = os.path.basename(filepath)
    module_name = os.path.splitext(filename)[0]
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
            
        # FIX: Patch relative imports to absolute for standalone execution
        # "from .nodes_dna import" -> "from nodes_dna import"
        code = re.sub(r'from \.(\w+) import', r'from \1 import', code)
        
        # "from . import X" -> "raise ImportError # from . import X" 
        # This triggers the except ImportError block usually present
        code = re.sub(r'from \. import', r'raise ImportError # from . import', code)
        
        # Create a module/namespace
        module = type(sys)(module_name)
        module.__file__ = filepath
        
        # Execute
        exec(code, module.__dict__)
        
        # Extract mappings
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            mapping = getattr(module, "NODE_CLASS_MAPPINGS")
            if not mapping:
                 print(f"Warning: {filename} has empty NODE_CLASS_MAPPINGS")
            return mapping, getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
            
    except Exception as e:
        print(f"ERROR loading {filename}: {e}")
        # informative trace
        # import traceback
        # traceback.print_exc()
        
    return {}, {}

def get_node_metadata():
    nodes = []
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Scan for nodes_*.py files
    node_files = [f for f in os.listdir(base_dir) if f.startswith("nodes_") and f.endswith(".py")]
    
    all_mappings = {}
    all_display_names = {}
    
    for f in node_files:
        path = os.path.join(base_dir, f)
        mappings, names = load_nodes_from_file(path)
        all_mappings.update(mappings)
        all_display_names.update(names)

    # Manually add missing ones if needed (like Mock ones)
    
    for class_name, cls in all_mappings.items():
        # Get basic info
        display_name = all_display_names.get(class_name, class_name)
        category = getattr(cls, "CATEGORY", "Uncategorized")
        description = getattr(cls, "DESCRIPTION", cls.__doc__ or "No description provided.")
        
        # Clean description
        if description:
            description = "\n".join([line.strip() for line in description.strip().splitlines() if line.strip()])
        
        # Inputs
        input_types = {}
        if hasattr(cls, "INPUT_TYPES"):
            try:
                input_types = cls.INPUT_TYPES()
            except:
                pass
                
        # Return types
        return_types = getattr(cls, "RETURN_TYPES", [])
        return_names = getattr(cls, "RETURN_NAMES", [])
        
        nodes.append({
            "class_name": class_name,
            "display_name": display_name,
            "category": category,
            "description": description,
            "inputs": input_types,
            "outputs": return_types,
            "output_names": return_names,
            "gpu": "GPU" in description or "Accelerated" in description or "Float32" in class_name
        })
        
    return nodes

def generate_html(nodes):
    # Categorize
    categories = {}
    for node in nodes:
        cat = node["category"].replace("FXTD Studios/Radiance/", "")
        cat_short = cat.split("/")[-1] # Take last part if nested
        # But maybe we want full hierarchy? Let's use clean simple names.
        
        # Custom mapping for cleaner TOC
        if "HDR" in cat: cat = "HDR Processing"
        elif "Color" in cat: cat = "Color Grading"
        elif "Film" in cat: cat = "Film Look"
        elif "Upscale" in cat: cat = "Upscaling"
        elif "Viewer" in cat: cat = "Viewer"
        elif "IO" in cat: cat = "IO & Utils"
        elif "Camera" in cat: cat = "Camera Simulation"
        elif "Resolution" in cat: cat = "Utilities"
        elif "Help" in cat: cat = "Utilities"
        elif "Prompt" in cat: cat = "Prompting"
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node)
        
    # Read Template (Head/Nav) hardcoded for speed/reliability
    
    html_start = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Node Reference - FXTD Studio Radiance</title>
    <!-- Use standard relative path for icon if exists -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #202020;
            --bg-card: rgba(35, 35, 35, 0.95);
            --bg-card-hover: rgba(45, 45, 45, 0.95);
            --bg-glass: rgba(255, 255, 255, 0.02);
            --text-primary: #e0e0e0;
            --text-secondary: #888888;
            --text-muted: #666666;
            --accent-gold: #888888;
            --accent-teal: #aaaaaa;
            --accent-violet: #888888;
            --border-default: rgba(255, 255, 255, 0.1);
            --border-hover: rgba(255, 255, 255, 0.2);
            --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.3);
            --shadow-glow: none;
            --gradient-brand: linear-gradient(135deg, #555 0%, #444 50%, #333 100%);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body { font-family: 'Inter', sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.7; min-height: 100vh; }
        .node-icon { filter: grayscale(100%); opacity: 0.7; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        
        /* Header */
        header { text-align: center; padding: 4rem 2rem; margin-bottom: 2rem; }
        header h1 { font-size: 2.75rem; font-weight: 800; margin-bottom: 0.5rem; color: #fff; }
        header .subtitle { font-size: 1.15rem; color: var(--text-secondary); margin-bottom: 1.5rem; }
        .badges { display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }
        .badge { padding: 0.5rem 1rem; border-radius: 50px; font-size: 0.8rem; font-weight: 600; border: 1px solid transparent; }
        .badge-gold { background: rgba(212, 168, 83, 0.15); border-color: rgba(212, 168, 83, 0.4); color: var(--accent-gold); }
        .badge-teal { background: rgba(45, 212, 191, 0.12); border-color: rgba(45, 212, 191, 0.3); color: var(--accent-teal); }
        .badge-violet { background: rgba(139, 92, 246, 0.12); border-color: rgba(139, 92, 246, 0.3); color: var(--accent-violet); }

        /* Nav */
        nav { background: #252525; border-bottom: 1px solid #333; padding: 0.75rem 2rem; position: sticky; top: 0; z-index: 100; }
        
        /* TOC */
        .toc { background: var(--bg-card); border-radius: 16px; padding: 2rem; margin-bottom: 2rem; border: 1px solid var(--border-default); }
        .toc h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; }
        .toc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
        .toc a { color: var(--text-secondary); text-decoration: none; font-size: 0.9rem; padding: 0.5rem; border-radius: 8px; transition: all 0.2s; }
        .toc a:hover { color: var(--accent-gold); background: var(--bg-glass); }

        /* Categories & Cards */
        .category { margin-bottom: 3rem; }
        .category-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-default); }
        .category-header h2 { font-size: 1.5rem; font-weight: 700; }
        .category-count { background: var(--bg-glass); padding: 0.35rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: var(--text-muted); border: 1px solid var(--border-default); }

        .nodes-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.25rem; }
        .node-card { background: var(--bg-card); border: 1px solid var(--border-default); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; transition: all 0.3s; position: relative; overflow: hidden; }
        .node-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-glow); transform: translateY(-4px); }
        .node-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--gradient-brand); opacity: 0; transition: opacity 0.35s ease; }
        .node-card:hover::before { opacity: 1; }

        .node-title { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
        .node-icon { font-size: 1.5rem; }
        .node-name { font-size: 1.2rem; font-weight: 700; color: var(--text-primary); }
        .node-class { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--accent-gold); background: rgba(212, 168, 83, 0.1); padding: 0.25rem 0.6rem; border-radius: 6px; display: inline-block; margin-bottom: 0.75rem; border: 1px solid rgba(212, 168, 83, 0.2); }
        .node-description { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem; line-height: 1.6; white-space: pre-wrap; }

        .node-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .tag { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.3rem 0.7rem; border-radius: 6px; background: var(--bg-glass); color: var(--text-muted); border: 1px solid var(--border-default); }
        .tag-gpu { background: rgba(45, 212, 191, 0.12); border-color: rgba(45, 212, 191, 0.25); color: var(--accent-teal); }
        .tag-pro { background: rgba(139, 92, 246, 0.12); border-color: rgba(139, 92, 246, 0.25); color: var(--accent-violet); }

        /* IO Table */
        .io-section { margin-top: 1rem; }
        .io-section h4 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border-default); }
        th { color: var(--text-muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; }
        td { color: var(--text-secondary); }
        td code { font-family: 'JetBrains Mono', monospace; color: var(--accent-gold); background: rgba(212, 168, 83, 0.1); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.8rem; }

        /* Search */
        .search-container { position: relative; margin-bottom: 2rem; }
        .search-container input { width: 100%; padding: 1rem 1rem 1rem 3rem; background: #252525; border: 1px solid #333; border-radius: 8px; color: #e0e0e0; font-size: 1rem; outline: none; }
        .search-icon { position: absolute; left: 1rem; top: 50%; transform: translateY(-50%); opacity: 0.5; }

        /* Footer */
        footer { text-align: center; padding: 3rem 2rem; margin-top: 2rem; border-top: 1px solid var(--border-default); }
        footer p { color: var(--text-muted); font-size: 0.9rem; }
    </style>
</head>
<body>
    <nav>
        <div style="max-width: 1440px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <a href="index.html" style="text-decoration: none; color: #e0e0e0; font-weight: 600; font-size: 1.2rem;">Radiance</a>
            <div style="display: flex; gap: 1rem;">
                <a href="index.html" style="color: #999; text-decoration: none; padding: 0.35rem 0.75rem;">Overview</a>
                <a href="node_reference.html" style="color: #fff; text-decoration: none; padding: 0.35rem 0.75rem; background: #444; border-radius: 4px;">Nodes</a>
                <a href="radiance_viewer.html" style="color: #999; text-decoration: none; padding: 0.35rem 0.75rem;">Viewer</a>
            </div>
        </div>
    </nav>

    <div class="container">
        <header>
            <h1>📚 Node Reference</h1>
            <p class="subtitle">Complete documentation for all FXTD Studio Radiance nodes</p>
            <div class="badges">
                <span class="badge badge-gold">v1.0.0</span>
                <span class="badge badge-teal">""" + str(len(nodes)) + """ Nodes</span>
                <span class="badge badge-violet">GPU Accelerated</span>
            </div>
        </header>

        <div class="search-container">
            <input type="text" id="nodeSearch" placeholder="Search nodes by name or class..." autocomplete="off">
            <span class="search-icon">🔍</span>
        </div>

        <nav class="toc">
            <h3>📋 Categories</h3>
            <div class="toc-grid">
"""
    
    # Generate TOC
    toc_html = ""
    for cat, items in sorted(categories.items()):
        toc_html += f'                <a href="#{cat.replace(" ", "_")}">{cat} ({len(items)})</a>\n'
    
    html_middle = """            </div>
        </nav>
"""

    # Generate Node Cards
    nodes_html = ""
    for cat, items in sorted(categories.items()):
        nodes_html += f"""
        <section class="category" id="{cat.replace(" ", "_")}">
            <div class="category-header">
                <h2>{cat}</h2>
                <span class="category-count">{len(items)} nodes</span>
            </div>
            <div class="nodes-grid">
"""
        for node in sorted(items, key=lambda x: x['display_name']):
            
            # Tags
            tags_html = ""
            if node['gpu']:
                tags_html += '<span class="tag tag-gpu">GPU</span>'
            if "Pro" in node['category']:
                tags_html += '<span class="tag tag-pro">Professional</span>'
            
            # Inputs table
            inputs_html = ""
            if node['inputs'] and 'required' in node['inputs']:
                inputs_html += """
                <div class="io-section">
                    <h4>Inputs</h4>
                    <table>
                        <tr><th>Name</th><th>Type</th><th>Default</th></tr>
"""
                for name, details in node['inputs']['required'].items():
                    dtype = details[0] if isinstance(details[0], str) else "COMBO"
                    default = str(details[1].get("default", "")) if len(details) > 1 and isinstance(details[1], dict) else ""
                    inputs_html += f"                        <tr><td>{name}</td><td><code>{dtype}</code></td><td>{default}</td></tr>\n"
                
                # Optional inputs
                if 'optional' in node['inputs']:
                    for name, details in node['inputs']['optional'].items():
                        dtype = details[0] if isinstance(details[0], str) else "COMBO"
                        inputs_html += f"                        <tr><td>{name} (opt)</td><td><code>{dtype}</code></td><td>-</td></tr>\n"
                        
                inputs_html += "                    </table></div>"

            # Outputs
            outputs_html = ""
            if node['outputs']:
                outputs_html += """
                <div class="io-section">
                    <h4>Outputs</h4>
                    <table>
                        <tr><th>Name</th><th>Type</th></tr>
"""
                # Handle return types being a tuple or list
                rets = node['outputs']
                names = node['output_names']
                
                for i, type_name in enumerate(rets):
                    name = names[i] if i < len(names) else type_name
                    outputs_html += f"                        <tr><td>{name}</td><td><code>{type_name}</code></td></tr>\n"
                outputs_html += "                    </table></div>"

            # Node Card HTML
            nodes_html += f"""
                <div class="node-card">
                    <div class="node-title">
                        <span class="node-icon">📦</span>
                        <span class="node-name">{node['display_name']}</span>
                    </div>
                    <code class="node-class">{node['class_name']}</code>
                    <p class="node-description">{node['description']}</p>
                    <div class="node-tags">{tags_html}</div>
                    {inputs_html}
                    {outputs_html}
                </div>
"""
        nodes_html += "            </div></section>"

    html_end = """
    </div>

    <footer>
        <p><strong>FXTD Studio Radiance</strong> &copy; 2026</p>
        <p>Professional Visual Effects Nodes for ComfyUI</p>
    </footer>

    <script>
        // Simple search functionality
        document.getElementById('nodeSearch').addEventListener('input', function(e) {
            const term = e.target.value.toLowerCase();
            const cards = document.querySelectorAll('.node-card');
            
            cards.forEach(card => {
                const text = card.textContent.toLowerCase();
                if(text.includes(term)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
            
            // Hide empty categories
            document.querySelectorAll('.category').forEach(cat => {
                const visible = cat.querySelectorAll('.node-card[style="display: block;"]').length > 0;
                const hasVisible = Array.from(cat.querySelectorAll('.node-card')).some(c => c.style.display !== 'none');
                cat.style.display = hasVisible ? 'block' : 'none';
            });
        });
    </script>
</body>
</html>
"""

    return html_start + toc_html + html_middle + nodes_html + html_end

if __name__ == "__main__":
    print(f"Generating docs from {os.getcwd()}...")
    nodes = get_node_metadata()
    print(f"Found {len(nodes)} nodes.")
    
    html = generate_html(nodes)
    
    output_path = os.path.join(os.path.dirname(__file__), "node_reference.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"Documentation generated at {output_path}")
