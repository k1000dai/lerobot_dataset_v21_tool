variable "secret_name" {
  description = "Name of the secret in AWS Secrets Manager"
  type        = string
}

variable "environment" {
  description = "Environment"
  type        = string
}

variable "system_name" {
  description = "System name"
  type        = string
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}
