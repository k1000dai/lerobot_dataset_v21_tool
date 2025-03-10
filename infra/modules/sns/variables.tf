variable "topic_name" {
  description = "Name of the SNS topic"
  type        = string
  default     = "data-pipeline-trigger"
}

variable "display_name" {
  description = "Display name of the SNS topic"
  type        = string
  default     = "Data Pipeline Trigger"
}

variable "lambda_function_arn" {
  description = "ARN of the Lambda function to subscribe to the topic"
  type        = string
  default     = null
}

variable "common_tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}