FROM python

WORKDIR /app

RUN apt-get update && \
    apt-get clean

RUN pip install --upgrade pip

COPY requirements.txt /app/


RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENTRYPOINT ["python", "main.py"]
CMD ["--run_mode", "training", "--dataset", "uci_credit_approval", "--model_type", "NN", "--imputing_type_numerical", "median", "--imputing_type_categorical", "mode", "--split_ratio", "0.7", "--filepath", "data/uci_credit_approval.csv"]
