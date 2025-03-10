variable "environment" {
  description = "Environment"
  type        = string
}

variable "system_name" {
  description = "System name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for the Batch compute environment"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "common_tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}
