FROM python:3.10.12-bookworm
# Pre-installed library and its version
RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install mlflow
RUN pip3 install seaborn
RUN pip3 install ppscore
RUN pip3 install dash
RUN pip3 install shap
RUN pip install dash_bootstrap_components

CMD tail -f /dev/null