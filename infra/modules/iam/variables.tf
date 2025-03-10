# Naming variables
variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
}

variable "system_name" {
  description = "System name"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "account" {
  description = "AWS account ID"
  type        = string
}


# Resource ARNs
variable "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret"
  type        = string
}


# Tags
variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}