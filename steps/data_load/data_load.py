# Databricks notebook source
# MAGIC %md
# MAGIC # Gemini - Data Loading

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

import os
import datetime

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication


def parse_arg(name):
  dbutils.widgets.get(name)
  arg = getArgument(name)
  print(f"Loaded argument {name}: {arg}")
  return arg


def filter_data(start_date, end_date, system, platform):
  sql_command= f"""SELECT substr(ev.tag_date, 1,7) month,ev.tag_name, ev.ts_utc, CAST(ev.tag_value AS DOUBLE) tag_value, ev.Status_code 
                   FROM gemini_sandbox.{platform} ev, gemini_sandbox.gemini_model_to_tag_mapping mt 
                   WHERE tag_date>= '{start_date}' AND tag_date <= '{end_date}' AND ev.tag_name = mt.TAG_NAME AND mt.Model = '{system}' 
                   ORDER BY ev.ts_utc """
  df = sqlContext.sql(sql_command)
  return df


def write_data(df, system, platform, environment="golden"):
  # parameters for paths
  system_name_clean = system_clean = system.replace(" ", "")
  timestamp = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S-%f")
  
  # create paths
  file_name = f"{system_clean}.parquet"
  container_path = os.path.join(environment, platform, system_clean, timestamp)
  folder_path = os.path.join("/mnt/gemini-dataprep", container_path)
  file_path = os.path.join(folder_path, file_name)
  
  # make directories
  os.makedirs(name=folder_path, exist_ok=True)
  print(folder_path)
  
  # write data
  df.repartition("tag_name").write.partitionBy("tag_name").parquet(file_path)
  return os.path.join(container_path, file_name)


def register_dataset(path, system, platform, environment, start_date, end_date, secret_scope, datastore_name="dataprep"):
  # TODO: move parameters to Azure Key Vault
  sp_auth = ServicePrincipalAuthentication(
      tenant_id=dbutils.secrets.get(scope=secret_scope, key="tenant_id"),
      service_principal_id=dbutils.secrets.get(scope=secret_scope, key="service_principal_id"),
      service_principal_password=dbutils.secrets.get(scope=secret_scope, key="service_principal_password")
  )
  ws = Workspace(
        subscription_id=parse_arg("--AZUREML_ARM_SUBSCRIPTION"),
        resource_group=parse_arg("--AZUREML_ARM_RESOURCEGROUP"),
        workspace_name=parse_arg("--AZUREML_ARM_WORKSPACE_NAME"),
        auth=sp_auth
  )
  datastore = Datastore(
    workspace=ws,
    name=datastore_name
  )
  file_dataset = Dataset.File.from_files(path=[(datastore, f"{path}/tag_name=*/*.parquet")])
  system_name_clean = system_clean = system.replace(" ", "")
  file_dataset = file_dataset.register(
    workspace=ws,
    name=f"{system_name_clean}",
    description=f"{system_name_clean} dataset",
    tags={"system": system, "platform": platform, "environment": environment, "start_date": start_date, "end_date": end_date},
    create_new_version=True
  )
  return file_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code execution

# COMMAND ----------

# Parse args
print("Parsing arguments")
environment = parse_arg("environment")
start_date = parse_arg("start_date")
end_date = parse_arg("end_date")
system = parse_arg("system")
platform = parse_arg("platform")
secret_scope = parse_arg("secret_scope")

# Load data
print("Loading data")
df = filter_data(
  start_date=start_date,
  end_date=end_date,
  system=system,
  platform=platform
)

# Write data
print("Write data")
path = write_data(
  df=df,
  system=system,
  platform=platform
)

# Register dataset
print("Registering dataset")
register_dataset(
  path=path,
  system=system,
  platform=platform,
  environment=environment,
  start_date=start_date,
  end_date=end_date
)
