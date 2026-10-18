"""Writes notebooks/run_app_in_colab.ipynb (upload widget + Streamlit app inside Colab through a Cloudflare tunnel)."""
import nbformat as nbf

INTRO = """# 🌿 Peatland pipeline - run the app in Colab
**Three cells, top to bottom (Runtime → Run all):**
1. **Get the project** - clones it from GitHub automatically (skipped if already unpacked). Only if the clone fails does an upload button appear: choose `peatland-landcover-dl.zip`.
2. **Install** - installs the packages (a few minutes the first time).
3. **Launch** - shows a link (**Option A**, through Colab itself); click it to open the app. A public backup link (Option B) is **off by default** - see the switch at the top of cell 3 and read its warning first. In the app, upload the CORINE file in the sidebar and press **Run pipeline**.

Tip: set **Runtime → Change runtime type → T4 GPU** first. Start with *demo* mode."""

UPLOAD = '''# 1) GET THE PROJECT - clone from GitHub; if that fails, an upload button appears for peatland-landcover-dl.zip
import glob, os, subprocess, zipfile
from pathlib import Path

REPO_URL = "https://github.com/SyedRafiAhmedGIS/peatland-landcover-dl.git"   # must be a PUBLIC repository

def _find_repo():
    hits = glob.glob("/content/**/app/streamlit_app.py", recursive=True)
    return str(Path(hits[0]).parent.parent) if hits else None

REPO = _find_repo()
if not REPO:
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/peatland-landcover-dl"], check=True, timeout=180,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})      # never wait for a password prompt
        REPO = _find_repo()
    except Exception as e:
        print("GitHub clone failed (", type(e).__name__, ") - falling back to the upload button.")
if REPO:
    print("Project ready at", REPO)
else:
    from google.colab import files
    print("Click 'Choose files' and select peatland-landcover-dl.zip ...")
    uploaded = files.upload()                      # <- upload button
    zips = [n for n in uploaded if n.lower().endswith(".zip")]
    if not zips:
        raise SystemExit("No .zip received - run this cell again and choose peatland-landcover-dl.zip")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall("/content")
    REPO = _find_repo()
    if not REPO:
        raise SystemExit("The zip does not contain app/streamlit_app.py - upload the full peatland-landcover-dl.zip")
os.chdir(REPO)
print("Ready. Working folder:", os.getcwd())'''

INSTALL = '''# 2) INSTALL (skips what Colab already has)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)
print("Packages installed.")
print("If Colab now shows a 'Restart session' banner, restart the runtime, then run cells 2 and 3 again (cell 1 is skipped automatically).")'''

LAUNCH = '''# 3) LAUNCH THE APP - two ways to open it (A is the most reliable)
import os, re, socket, subprocess, time, urllib.request
from IPython.display import HTML, display

# Option B publishes the app on the open internet WITHOUT a password, with XSRF/CORS protection switched off, and downloads an
# unpinned, unverified cloudflared binary. Anyone who gets the URL can use your session and upload files to it.
# Leave this False unless Option A does not work for you.
ENABLE_PUBLIC_TUNNEL = False

subprocess.run(["pkill", "-f", "streamlit run"], check=False)      # makes re-running this cell safe (frees port 8501)
time.sleep(1)
app = subprocess.Popen(["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.headless", "true",
                        "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
                       stdout=open("app.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):                                   # wait until the app answers
    try:
        urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("The app did not start - open app.log in the Files panel to see why.")
print("App is running inside Colab.")

# Option A - Colab's own proxy (no external DNS needed): click the link it shows
try:
    from google.colab import output
    output.serve_kernel_port_as_window(8501)
    print("OPTION A: click the link above to open the app in a new tab.")
except Exception as e:
    print("Option A unavailable:", type(e).__name__)

# Option B - public Cloudflare tunnel (backup; some networks resolve it slowly or not at all)
url = None
if ENABLE_PUBLIC_TUNNEL:
    try:
        subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                        "-O", "cloudflared"], check=True)
        os.chmod("cloudflared", 0o755)
        tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501"],
                                  stdout=subprocess.DEVNULL, stderr=open("tunnel.log", "w"))
        for _ in range(90):                               # wait for the tunnel to be really connected, not just named
            log = open("tunnel.log").read()
            m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", log)
            if m and "Registered tunnel connection" in log:
                try:
                    socket.getaddrinfo(m.group(0).split("//")[1], 443)
                    time.sleep(15)                        # let the DNS record spread before showing the link
                    url = m.group(0); break
                except OSError:
                    pass
            time.sleep(1)
    except Exception as e:
        print("Option B (tunnel) unavailable:", type(e).__name__)
    if url:
        display(HTML(f'<div style="padding:14px;border-radius:12px;background:#e0f2fe;font-family:Segoe UI,Arial,sans-serif">'
                     f'<b>OPTION B (backup link):</b> <a href="{url}" target="_blank">{url}</a>'
                     f'<div style="color:#475569;font-size:13px;margin-top:4px">If it says "site can\\'t be reached", wait 1 minute and refresh, '
                     f'or use Option A. Anyone with this link can use your session - do not share it.</div></div>'))
    elif os.path.exists("tunnel.log"):
        print("Option B: the tunnel did not connect - use Option A (see tunnel.log in the Files panel).")
else:
    print("Option B (public tunnel) is switched off - use the Option A link above.")
print("Keep this notebook running while you use the app.")'''

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(UPLOAD), nbf.v4.new_code_cell(INSTALL),
            nbf.v4.new_code_cell(LAUNCH)]
nb.metadata = {"colab": {"name": "run_app_in_colab.ipynb"}, "accelerator": "GPU",
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, "notebooks/run_app_in_colab.ipynb")
