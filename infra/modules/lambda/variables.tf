variable "environment" {
  description = "Environment"
  type        = string
}

variable "system_name" {
  description = "System name"
  type        = string
}

variable "lambda_sources" {
  description = "List of Lambda source code objects"
  type = list(object({
    path     = string
    filename = string
  }))
  default = []
}

variable "handler" {
  description = "Lambda function handler"
  type        = string
  default     = "lambda_handler"
}

variable "timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 180
}

variable "memory_size" {
  description = "Lambda function memory size in MB"
  type        = number
  default     = 128
}

variable "secret_name" {
  description = "Name of the secret in AWS Secrets Manager"
  type        = string
}

variable "lerobot_bucket_name" {
  description = "Name of the LeRobot output bucket"
  type        = string
}

variable "batch_job_queue" {
  description = "AWS Batch job queue name"
  type        = string
}

variable "batch_job_definition" {
  description = "AWS Batch job definition name"
  type        = string
}

variable "execution_role_arn" {
  description = "ARN of the IAM execution role for the Lambda function"
  type        = string
}

variable "common_tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}
