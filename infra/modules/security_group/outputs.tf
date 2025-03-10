output "batch_security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.batch_security_group.id
}

output "vpc_endpoint_security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.vpc_endpoint_security_group.id
}
