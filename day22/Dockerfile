# ============================================================
# Docker vs Virtual Machines:
# VM: har app ka apna Guest OS hota hai + Hypervisor overhead
# Docker: sab apps ek hi Host OS kernel share karte hain
#         sirf app + libs isolate hoti hain (namespaces)
#
# Linux Namespaces:
# - PID namespace: har container ke apne process IDs
# - NET namespace: apna network interface aur ports
# - IPC namespace: apna inter-process communication
#
# Layer Caching:
# - requirements.txt alag copy karte hain pehle
# - agar code change ho toh sirf code layer rebuild hoti hai
# - dependencies O(1) time mein rebuild (cached)
# - bina caching ke O(N) time lagta har build mein
# ============================================================

FROM python:3.11-slim

# working directory
WORKDIR /app

# copy requirements first (layer caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY main.py .
COPY loan_approval_model.pkl .

# create non-root user for security
RUN useradd -m appuser
USER appuser

# expose port
EXPOSE 8000

# start fastapi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
