# Security group for Batch compute environment
resource "aws_security_group" "batch_security_group" {
  name   = "${var.environment}-${var.system_name}-sg-batch-compute-env"
  vpc_id = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.system_name}-sg-batch-compute-env"
  })
}


resource "aws_security_group" "vpc_endpoint_security_group" {
  name   = "${var.environment}-${var.system_name}-sg-vpc-endpoint"
  vpc_id = var.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.system_name}-sg-vpc-endpoint"
  })
}
