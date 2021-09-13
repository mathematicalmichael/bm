FROM python:3.9.3

WORKDIR /app

EXPOSE 8080

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN find /usr/local/lib/python3.9/site-packages/streamlit -type f \( -iname \*.py -o -iname \*.js \) -print0 | xargs -0 sed -i 's/healthz/health-check/g'

COPY main.py ./main.py

CMD streamlit run --server.port 8080 --browser.serverAddress 0.0.0.0 --server.enableCORS False --server.enableXsrfProtection False main.py


